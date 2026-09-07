get_top_level_symbol(expr::Symbol) = expr
function get_top_level_symbol(expr::Expr)
    return if Meta.isexpr(expr, :ref)
        get_top_level_symbol(expr.args[1])
    elseif Meta.isexpr(expr, :.)
        get_top_level_symbol(expr.args[1])
    else
        _issue_error("get_top_level_symbol: unhandled expression $(expr)")
    end
end

# VarNames.varname(:(xs[i])) returns something like
#     VarName{:xs}(Index((esc(i),), ...))
# which gets `i` from the caller's scope. However, when we insert this expression into a
# larger Expr (as in `@wren` and its RGF), we want to take `i` from the scope of the larger
# Expr itself, rather than the caller's scope. That necessitates removing the `esc()` call.
remove_escapes(x) = x
function remove_escapes(expr::Expr)
    expr.head == :escape && return remove_escapes(only(expr.args))
    return Expr(expr.head, map(remove_escapes, expr.args)...)
end

assign_or_set!!(lhs::Symbol, rhs, vn) = :($lhs = $rhs)
function assign_or_set!!(lhs::Expr, rhs, vn)
    left_top_sym = get_top_level_symbol(lhs)
    return :(
        $left_top_sym = $(Accessors.set)(
            $left_top_sym,
            $(VarNames.with_mutation)($(VarNames.getoptic)($vn)),
            $rhs,
        )
    )
end

emit_model_expr(expr, accs, conditioned, latent_source, parameter_space) = expr
function emit_model_expr(stmt::WrenExpr, accs, conditioned, latent_source, parameter_space)
    args = map(
        arg -> emit_model_expr(arg, accs, conditioned, latent_source, parameter_space),
        stmt.args,
    )
    return Expr(stmt.head, args...)
end
function emit_model_expr(
    stmt::OtherStatement,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    return stmt.expr
end

function emit_model_expr(stmt::Sample, accs, conditioned, latent_source, parameter_space)
    @gensym rhs val tval init_tval logjac vn child_accs child_params
    rhs_expr = emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space)
    vn_expr = remove_escapes(varname(stmt.lhs, false))
    template = get_top_level_symbol(stmt.lhs)
    return quote
        $rhs = $rhs_expr
        $vn = $vn_expr
        if $(Wren.is_submodel)(typeof($rhs))
            $val, $child_params, $child_accs =
                $(evaluate_submodel)($rhs, $latent_source, $parameter_space, $accs, $vn)
            $(assign_or_set!!(stmt.lhs, child_params, vn))
            $accs = $(combine_with_namespace)(
                $accs,
                $child_accs,
                $vn,
                $(stmt.lhs isa Symbol ? nothing : template),
            )
        else
            if $(haskey)($conditioned, $vn)
                $val = $(getindex)($conditioned, $vn)
                $(assign_or_set!!(stmt.lhs, val, vn))
                $accs = $(accumulate_observe!!)(
                    $accs,
                    $val,
                    $vn,
                    $rhs,
                    $template,
                    $(Val{stmt.name}()),
                )
            else
                $init_tval = $(get_latent)($latent_source, $vn, $rhs, $(Val{stmt.name}()))
                $val, $tval, $logjac =
                    $(apply_parameter_space)($parameter_space, $init_tval, $vn, $rhs)
                $(assign_or_set!!(stmt.lhs, val, vn))
                $accs = $(accumulate_assume!!)(
                    $accs,
                    $val,
                    $tval,
                    $vn,
                    $rhs,
                    $logjac,
                    $template,
                    $(Val{stmt.name}()),
                )
            end
        end
        # A tilde expression evaluates to the sampled value or submodel return value.
        $val
    end
end

# `@iid lhs ~ rhs` (or the `@observe` variants) means `lhs ~ iid(rhs, size(lhs))`: one
# array-valued distribution of independent draws from `rhs`, with the same shape as the
# current value of `lhs`. We rewrite the wrapped tilde-statement's right-hand side and then
# emit it exactly as we would a plain `Sample` or `Observe`.
function emit_model_expr(stmt::IID, accs, conditioned, latent_source, parameter_space)
    tilde = stmt.tilde
    rhs_expr = emit_model_expr(tilde.rhs, accs, conditioned, latent_source, parameter_space)
    iid_rhs = :($(iid)($rhs_expr, $(Base.size)($(tilde.lhs))))
    return emit_model_expr(
        _with_rhs(tilde, iid_rhs),
        accs,
        conditioned,
        latent_source,
        parameter_space,
    )
end
_with_rhs(stmt::Sample, rhs) = Sample(stmt.name, stmt.lhs, rhs)
_with_rhs(stmt::Observe, rhs) = Observe(stmt.name, stmt.lhs, rhs)

# @plate for i in 1:N
#    y[i] ~ Normal(x[i])
# end
#
# CURRENTLY gets lowered to something like:
#
# y[1:N] ~ independent([Normal(x[i]) for i in 1:N])
#
# This can be improved! https://github.com/JuliaBayes/ProbabilityMeasures.jl/issues/15
function emit_model_expr(stmt::Plate, accs, conditioned, latent_source, parameter_space)
    # stmt.body isa block expr
    @gensym loop_iters
    initialise_iter_expr = :($loop_iters = $(stmt.loop_iters))
    body_exprs = [
        vectorise_plate_stmt(
            s,
            stmt.loop_idx,
            loop_iters,
            accs,
            conditioned,
            latent_source,
            parameter_space,
        ) for s in stmt.body.args
    ]
    return Expr(:block, initialise_iter_expr, body_exprs...)
end

"""
    Indexed

Marker struct to indicate that the argument is an object indexed by the plate loop index.
"""
struct Indexed{T}
    data::T
end

"""
    independent(dists::AbstractArray)

Return a distribution over arrays with the same shape as `dists`, where each entry
is independently drawn from `dists[i]`.

Used by `@plate` for indexed distributions without a specialised `batched` method.
Backends (e.g. the Distributions and ProbabilityMeasures extensions) add methods for
their own distribution types.
"""
function independent end

"""
    iid(dist, dims::Dims)

Return independent draws from `dist` in an array of shape `(size(dist)..., dims...)`.
The distribution is stored once. `@plate` uses this when the distribution does not
depend on the loop index.

```julia
@wren function f(N)
    x ~ iid(Normal(), (N,))   # length N
    W ~ iid(Normal(), (3, N)) # 3 × N
end
```

Inside `@wren`, `@iid y ~ dist` is shorthand for `y ~ iid(dist, size(y))`.

The fallback is `independent(Fill(dist, dims))`; backends may add specialised methods.
"""
iid(dist, dims::Dims) = independent(FillArrays.Fill(dist, dims))

# TODO(penelopeysm): I'm unsure about type stability here because of the vararg
# specialisation thing. We might need the fallback on generated functions with enumerated
# arguments
function batched(iters::I, ::Val{F}, args...; kwargs...) where {I,F}
    # Default fallback for batching-unaware calls: use independent instead
    return independent([
        F((a isa Indexed ? a.data[i] : a for a in args)...; kwargs...) for i in iters
    ])
end

function make_product_expr(rhs, loop_idx, loop_iters)
    # This returns `Wren.independent([$rhs for $loop_idx in $loop_iters])`.
    Expr(
        :call,
        Wren.independent,
        Expr(:comprehension, Expr(:generator, rhs, Expr(:(=), loop_idx, loop_iters))),
    )
end
function occurs_in_kwargs(parameters::Expr, loop_idx::Symbol)
    for kw in parameters.args
        if kw isa Symbol
            if kw == loop_idx
                # f(...; loop_idx) i.e., shorthand for f(...; loop_idx=loop_idx)
                return true
            else
                continue
            end
        elseif kw isa Expr && kw.head == :kw
            # f(...; kw = val_expr)
            occurs_in_expr(kw.args[2], loop_idx) && return true
        else
            error("@plate: encountered unexpected kwarg expression $(kw)")
        end
    end
    return false
end

function batch_expr(expr, loop_idx, loop_iters)
    # Check if the expr contains the loop index anywhere.
    return if !occurs_in_expr(expr, loop_idx)
        # If not, it's a constant wrt the loop.
        #
        # TODO(penelopeysm): Use `iid(rhs, size(loop_iters))` here once it's in main,
        # instead of `batched`. https://github.com/JuliaBayes/Wren.jl/pull/71
        Expr(:call, Wren.batched, loop_iters, expr)
    else
        if expr isa Expr && expr.head == :call
            # First we'll check if the call has kwargs and if the loop index is referenced
            # in any of them. If so, we'll fall back gracefully to the product distribution,
            if (
                length(expr.args) >= 2 &&
                expr.args[2] isa Expr &&
                expr.args[2].head == :parameters &&
                occurs_in_kwargs(expr.args[2], loop_idx)
            )
                make_product_expr(expr, loop_idx, loop_iters)
            else
                # batch the call e.g. f(x[i], y[i])
                #   --> batched(1:J, Val{f}(), (Indexed(x[1:J]), Indexed(y[1:J])))
                #
                # Note that we have to use `Val` here because `f` is often a type constructor
                # e.g. `Normal`, which prevents us from dispatching on `::typeof(f)` (since
                # typeof(Normal) is UnionAll). We use `Val{f}()` rather than `Val(f)` to make
                # sure it's type stable.
                fname = Expr(:call, Expr(:curly, Base.Val, expr.args[1]))

                # Check if there are kwargs. If so, we extract them first.
                kwargs_expr, positional_args =
                    if length(expr.args) >= 2 &&
                       expr.args[2] isa Expr &&
                       expr.args[2].head == :parameters
                        expr.args[2], expr.args[3:end]
                    else
                        nothing, expr.args[2:end]
                    end

                batched_positional_args = map(
                    arg -> begin
                        if arg isa Expr && arg.head == :parameters
                            # Keyword arguments; we'll forward these directly into batched,
                            # since we ascertained earlier that they don't contain the loop
                            # index.
                            arg
                        else
                            # Positional arguments.
                            expanded, _ = replace_loop_idx_with_iters(
                                arg,
                                loop_idx,
                                loop_iters,
                            )
                            # TODO(penelopeysm): Should we be using the second return value here to
                            # determine whether or not to view???
                            if occurs_in_expr(arg, loop_idx)
                                viewed =
                                    if expanded isa Expr && expanded.head == :ref
                                        Expr(:macrocall, Symbol("@view"), nothing, expanded)
                                    else
                                        expanded
                                    end
                                Expr(:call, Indexed, viewed)
                            else
                                expanded
                            end
                        end
                    end,
                    positional_args,
                )

                if kwargs_expr === nothing
                    Expr(:call, Wren.batched, loop_iters, fname, batched_positional_args...)
                else
                    Expr(
                        :call,
                        Wren.batched,
                        kwargs_expr,
                        loop_iters,
                        fname,
                        batched_positional_args...,
                    )

                end
            end
        else
            # A form we don't know how to handle, but it does contain the loop index in some
            # form. Just wrap in a product
            make_product_expr(expr, loop_idx, loop_iters)
        end
    end
end

"""
    replace_loop_idx_with_iters(expr, loop_idx::Symbol, loop_iters)

Replace all occurrences of `loop_idx` in `expr` with `loop_iters` and broadcast where necessary.

Returns a tuple of `(new_expr, is_batched)` where `new_expr` is the modified expression and
`is_batched` indicates whether or not the expression was batched (i.e., it used to refer to
a loop index but now refers to the iteration range).
"""
function replace_loop_idx_with_iters(expr, loop_idx::Symbol, loop_iters)
    return if expr isa Symbol
        expr == loop_idx ? (loop_iters, true) : (expr, false)
    elseif expr isa QuoteNode
        expr, false
    elseif expr isa Expr
        if expr.head in (:ref, :., :call)
            new_args_and_is_batched = map(
                arg -> replace_loop_idx_with_iters(arg, loop_idx, loop_iters),
                expr.args,
            )
            new_args = first.(new_args_and_is_batched)
            any_arg_was_batched = any(last.(new_args_and_is_batched))
            if expr.head == :call && any_arg_was_batched
                # e.g. i + 1 --> apart from just replacing i, we also need to broadcast the
                # operation `i .+ 1`.
                new_expr = if Meta.isoperator(expr.args[1])
                    # Operators like `x + y` need to become `.+(x, y)`
                    broadcasted_op = Symbol(".", expr.args[1])
                    Expr(expr.head, broadcasted_op, new_args[2:end]...)
                else
                    # Functions like `f(x, y)` need to become `f.(x, y)`. Annoyingly the
                    # requisite Expr has a completely different form from the operator
                    # case........
                    func_sym = new_args[1]
                    func_args = new_args[2:end]
                    Expr(Symbol("."), func_sym, Expr(:tuple, func_args...))
                end
                return new_expr, true
            elseif expr.head == :ref
                # e.g. x[i + 1] -> x[@. (1:J) + 1] also causes the overall indexing
                # expression to be batched
                new_expr = Expr(expr.head, new_args...)
                new_expr, any_arg_was_batched
            else
                # e.g. x[const] -> x[const]
                new_expr = Expr(expr.head, new_args...)
                new_expr, false
            end
        else
            error("@plate: unsupported expression $(expr) with head $(expr.head)")
        end
    else
        # e.g. literals
        expr, false
    end
end

vectorise_plate_stmt(lnn::LineNumberNode, idx, iters, ax, cond, ls, ps) = lnn
function vectorise_plate_stmt(::IID, idx, iters, ax, cond, ls, ps)
    # `@iid` already produces one array-valued statement; batching it over a plate has no
    # defined semantics yet.
    return error("@iid is not supported inside @plate")
end
function vectorise_plate_stmt(
    stmt::Sample,
    loop_idx,
    loop_iters,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    new_lhs, _ = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters)
    rhs_expr = _wren_expr_to_model_expr(stmt.rhs)
    new_rhs = batch_expr(rhs_expr, loop_idx, loop_iters)
    return emit_model_expr(
        Sample(stmt.name, new_lhs, new_rhs),
        accs,
        conditioned,
        latent_source,
        parameter_space,
    )
end

occurs_in_expr(expr::Symbol, loop_idx::Symbol) = expr == loop_idx
occurs_in_expr(expr::QuoteNode, loop_idx::Symbol) = false
occurs_in_expr(expr::Expr, loop_idx::Symbol) =
    any(arg -> occurs_in_expr(arg, loop_idx), expr.args)
occurs_in_expr(expr, loop_idx) = false

function vectorise_plate_stmt(
    stmt::Observe,
    loop_idx,
    loop_iters,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    new_lhs, _ = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters)
    rhs_expr = _wren_expr_to_model_expr(stmt.rhs)
    new_rhs = batch_expr(rhs_expr, loop_idx, loop_iters)
    return emit_model_expr(
        Observe(stmt.name, new_lhs, new_rhs),
        accs,
        conditioned,
        latent_source,
        parameter_space,
    )
end

function _maybe_view(v::AbstractArray, idxs...)
    return if ndims(view(v, idxs...)) == 0
        v[idxs...]
    else
        view(v, idxs...)
    end
end

function emit_model_expr(
    stmt::AddLogLikelihood,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    @gensym value
    value_expr =
        emit_model_expr(stmt.expr, accs, conditioned, latent_source, parameter_space)
    return quote
        if $(hasacc)($accs, $(Val{LOG_LIKELIHOOD_ACCNAME}()))
            $value = $value_expr
            $accs = $(accumulate_addloglikelihood!!)($accs, $value, $(Val{stmt.name}()))
        end
    end
end

function emit_model_expr(stmt::Observe, accs, conditioned, latent_source, parameter_space)
    @gensym rhs val vn
    rhs_expr = emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space)
    vn_expr = remove_escapes(varname(stmt.lhs, false))
    template = get_top_level_symbol(stmt.lhs)
    lhs_val = if stmt.lhs isa Expr && stmt.lhs.head == :ref
        Expr(:call, _maybe_view, stmt.lhs.args...)
    else
        stmt.lhs
    end
    return quote
        $val = $(lhs_val)
        $rhs = $rhs_expr
        $vn = $vn_expr
        $accs =
            $(accumulate_observe!!)($accs, $val, $vn, $rhs, $template, $(Val{stmt.name}()))
    end
end

function emit_model_expr(
    stmt::ReturnStatement,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    body = Expr(:block)
    @gensym retval
    if stmt.expr !== nothing
        return_expr =
            emit_model_expr(stmt.expr, accs, conditioned, latent_source, parameter_space)
        push!(body.args, :($retval = $return_expr))
    else
        push!(body.args, :($retval = nothing))
    end
    push!(body.args, :(return ($retval, $accs)))
    return body
end

function emit_statements(stmts, accs, conditioned, latent_source, parameter_space)
    body = Expr(:block)
    for stmt in stmts
        push!(
            body.args,
            emit_model_expr(stmt, accs, conditioned, latent_source, parameter_space),
        )
    end
    return body
end
