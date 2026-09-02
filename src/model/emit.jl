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
    # Note: this mutates in place, which is only safe because `make_evaluation_fn` copies
    # every argument that is written to at the top of the model body. Mutating the model's
    # own argument array would be undefined behaviour under Enzyme, which is handed that
    # array as a constant: storing active values into constant memory and reading them back
    # silently drops the gradient of everything downstream (e.g. an `@observe` inside a
    # `@plate` that depends on a variable sampled in the same plate).
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

# @plate for i in 1:N
#    y[i] ~ Normal(x[i])
# end
#
# CURRENTLY gets lowered to something like:
#
# y[1:N] ~ _product([Normal(x[i]) for i in 1:N])
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

# Semantically equivalent to product_distribution, but we can overload per distribution
# backend
function _product end

# Q: unsure about type stability here because of the vararg specialisation thing. 
# Might need the fallback on generated functions.
function batched(iters::I, ::Val{F}, args...) where {I,F}
    # Default fallback for batching-unaware calls: use _product instead
    return _product([F((a isa Indexed ? a.data[i] : a for a in args)...) for i in iters])
end

function batch_expr(expr, loop_idx, loop_iters)
    # Check if the expr contains the loop index anywhere. If not, it's a constant
    # wrt the loop, so we can do a simple broadcast
    return if occurs_in_expr(expr, loop_idx)
        if expr isa Expr && expr.head == :call
            # batch the call e.g. f(x[i], y[i])
            #   --> batched(1:J, Val{f}(), (Indexed(x[1:J]), Indexed(y[1:J])))
            #
            # Note that we have to use `Val` here because `f` is often a type constructor
            # e.g. `Normal`, which prevents us from dispatching on `::typeof(f)` (since
            # typeof(Normal) is UnionAll). We use `Val{f}()` rather than `Val(f)` to make
            # sure it's type stable.
            fname = Expr(:call, Expr(:curly, Base.Val, expr.args[1]))
            fargs = map(
                arg -> begin
                    expanded = replace_loop_idx_with_iters(arg, loop_idx, loop_iters)
                    if occurs_in_expr(arg, loop_idx)
                        viewed = if expanded isa Expr && expanded.head == :ref
                            Expr(:macrocall, Symbol("@view"), nothing, expanded)
                        else
                            expanded
                        end
                        Expr(:call, Indexed, viewed)
                    else
                        expanded
                    end
                end,
                expr.args[2:end],
            )
            Expr(:call, Wren.batched, loop_iters, fname, fargs...)
        else
            # a form we don't know how to handle -- just wrap in a product
            #  --> _product([$expr for i in 1:J])
            Expr(
                :call,
                Wren._product,
                Expr(
                    :comprehension,
                    Expr(:generator, expr, Expr(:(=), loop_idx, loop_iters)),
                ),
            )
        end
    else
        # Constant wrt loop index, so no special handling needed
        #  --> batched(1:J, $expr)
        Expr(:call, Wren.batched, loop_iters, expr)
    end
end

"""
    replace_loop_idx_with_iters(expr, loop_idx::Symbol, loop_iters)

Replace all occurrences of `loop_idx` in `expr` with `loop_iters`.
"""
function replace_loop_idx_with_iters(expr, loop_idx::Symbol, loop_iters)
    return if expr isa Symbol
        expr == loop_idx ? loop_iters : expr
    elseif expr isa QuoteNode
        QuoteNode(replace_loop_idx_with_iters(expr.value, loop_idx, loop_iters))
    elseif expr isa Expr
        if expr.head in (:ref, :., :call)
            new_args = map(
                arg -> replace_loop_idx_with_iters(arg, loop_idx, loop_iters),
                expr.args,
            )
            new_expr = Expr(expr.head, new_args...)
            if expr.head == :call && any(arg -> arg == loop_idx, expr.args)
                # e.g. i + 1 --> apart from just replacing i, we also need to broadcast the
                # operation `@. (1:J) + 1`. Note that we need the `nothing` because
                # `:macrocall` expects a LineNumberNode, but we can't provide it.
                Expr(:macrocall, Symbol("@__dot__"), nothing, new_expr)
            else
                # ok to just replace.
                new_expr
            end
        else
            error("@plate: unsupported expression $(expr) with head $(expr.head)")
        end
    else
        # e.g. literals
        expr
    end
end

vectorise_plate_stmt(lnn::LineNumberNode, idx, iters, ax, cond, ls, ps) = lnn
function vectorise_plate_stmt(
    stmt::Sample,
    loop_idx,
    loop_iters,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    new_lhs = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters)
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
occurs_in_expr(expr::QuoteNode, loop_idx::Symbol) = occurs_in_expr(expr.value, loop_idx)
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
    new_lhs = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters)
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
