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


# In submodel.jl we override this when `val` is an `UncollectedParams` sentinel, so that we
# don't ever actually try to set it into the parent object, which can cause type instability
# issues (and consequently Enzyme errors) if the parent object is e.g. an array with a
# eltype that `UncollectedParams` doesn't subtype.
set_lhs!!(obj, optic, val) = Accessors.set(obj, optic, val)

set_namespace_copy(obj, optic, val) = Accessors.set(obj, optic, val)
# A deconditioned child namespace may be absent from the observed record.
set_namespace_copy(obj::NamedTuple, ::VarNames.Property{sym}, val) where {sym} =
    merge(obj, NamedTuple{(sym,)}((val,)))
function set_namespace_copy(obj::Array{T}, optic::VarNames.Index, val::V) where {T,V}
    U = Union{T,V}
    if Base.isbitsunion(U) && all(i -> i isa Union{Integer,CartesianIndex}, optic.ix)
        # For example, replacing (y=1.0,) with VarNamedTuple(a=0.5) should use
        # Union{typeof((y=1.0,)),typeof(VarNamedTuple(a=0.5))}, not Any. Accessors.set
        # and BangBang.setindex!! use promote_type here, which widens these records to
        # Any and boxes the remaining observed entries. Copy before the first write:
        # BangBang.setindex!! alone can also mutate a shared array when its eltype fits.
        result = copyto!(similar(obj, U), obj)
        Base.setindex!(result, val, optic.ix...; optic.kw...)
        return result
    end
    return Accessors.set(obj, optic, val)
end

# Copy a shared namespace container at its first write, preserving observed siblings.
# Later writes can mutate the owned container without copying it again.
@inline function set_namespace_lhs!!(obj, observed, optic, val)
    head, tail = VarNames.ohead(optic), VarNames.otail(optic)
    inner = if tail isa VarNames.Iden
        val
    else
        set_namespace_lhs!!(
            _scope_conditioned(obj, head),
            _scope_conditioned(observed, head),
            tail,
            val,
        )
    end
    return if obj isa NamedTuple || obj === observed
        set_namespace_copy(obj, head, inner)
    else
        Accessors.set(obj, VarNames.with_mutation(head), inner)
    end
end

assign_or_set!!(lhs::Symbol, rhs, vn) = :($lhs = $rhs)
function assign_or_set!!(lhs::Expr, rhs, vn)
    left_top_sym = get_top_level_symbol(lhs)
    return :(
        $left_top_sym = $(set_lhs!!)(
            $left_top_sym,
            $(VarNames.with_mutation)($(VarNames.getoptic)($vn)),
            $rhs,
        )
    )
end

emit_model_expr(expr, accs, conditioned, latent_source, parameter_space, ctx) = expr
function emit_model_expr(
    stmt::WrenExpr,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    args = map(
        arg -> emit_model_expr(arg, accs, conditioned, latent_source, parameter_space, ctx),
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
    ctx,
)
    return stmt.expr
end

function emit_model_expr(
    stmt::Sample,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    @gensym rhs val tval init_tval logjac vn child_accs child_params scoped_args
    rhs_expr =
        emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space, ctx)
    vn_expr = remove_escapes(varname(stmt.lhs, false))
    template = get_top_level_symbol(stmt.lhs)
    collect_submodel_params =
        stmt.lhs_is_reused ? CollectSubmodelParams() : DiscardSubmodelParams()
    assign_child_params = assign_or_set!!(stmt.lhs, child_params, vn)
    if stmt.lhs isa Expr
        # This generator only has a symbol for `conditioned`: the cached evaluation
        # function is shared by condition/decondition and parent-supplied overrides.
        # Julia folds this key check when specialising on the NamedTuple's type.
        assign_child_params = quote
            if $(haskey)($conditioned, $(QuoteNode(template)))
                $template = $(set_namespace_lhs!!)(
                    $template,
                    $(getproperty)($conditioned, $(QuoteNode(template))),
                    $(VarNames.getoptic)($vn),
                    $child_params,
                )
            else
                $assign_child_params
            end
        end
    end
    return quote
        $rhs = $rhs_expr
        $vn = $vn_expr
        if $(WrenCore.is_submodel)(typeof($rhs))
            if $rhs isa $(Model)
                # Scope before recursive evaluation can widen the site's VarName.
                $scoped_args = $(_scope_submodel)(
                    $rhs,
                    $latent_source,
                    $parameter_space,
                    $accs,
                    $conditioned,
                    $vn,
                )
                $val, $child_params, $child_accs = $(_evaluate_scoped_submodel)(
                    $rhs,
                    $scoped_args...,
                    $collect_submodel_params,
                    $ctx,
                )
            else
                $val, $child_params, $child_accs = $(evaluate_submodel)(
                    $rhs,
                    $latent_source,
                    $parameter_space,
                    $accs,
                    $conditioned,
                    $vn,
                    $collect_submodel_params,
                    $ctx,
                )
            end
            $assign_child_params
            $accs = $(combine_from_scope)(
                $accs,
                $child_accs,
                $vn,
                $(stmt.lhs isa Symbol ? nothing : template),
            )
        else
            # Whole-symbol membership is known from the NamedTuple's type.
            if $(haskey)($conditioned, $(QuoteNode(template)))
                $val = $(VarNames.getvalue)($conditioned, $vn)
                # Nothing reads an observed element back unless the symbol is reused. A
                # plain symbol is always bound, since it is the template below.
                $(
                    stmt.lhs_is_reused || stmt.lhs isa Symbol ?
                    assign_or_set!!(stmt.lhs, val, vn) : nothing
                )
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

function emit_model_expr(stmt::IID, accs, conditioned, latent_source, parameter_space, ctx)
    tilde = stmt.tilde
    rhs_expr =
        emit_model_expr(tilde.rhs, accs, conditioned, latent_source, parameter_space, ctx)
    dims_expr = stmt.size === nothing ? :($(Base.size)($(tilde.lhs))) : stmt.size
    iid_rhs = :($(iid)($rhs_expr, $dims_expr))
    return emit_model_expr(
        _with_rhs(tilde, iid_rhs),
        accs,
        conditioned,
        latent_source,
        parameter_space,
        ctx,
    )
end
_with_rhs(stmt::Sample, rhs) =
    Sample(stmt.name, stmt.lhs, rhs, stmt.lhs_top_level_symbol, stmt.lhs_is_reused)
_with_rhs(stmt::Observe, rhs) = Observe(stmt.name, stmt.lhs, rhs)

function emit_model_expr(
    stmt::Batch,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    # stmt.body isa block expr
    @gensym loop_iters loop_iters_size
    initialise_iter_expr = :($loop_iters = $(stmt.loop_iters))
    initialise_size_expr = :($loop_iters_size = $(Base.size)($loop_iters))
    body_exprs = [
        vectorise_batch_stmt(
            s,
            stmt.loop_idx,
            loop_iters,
            loop_iters_size,
            accs,
            conditioned,
            latent_source,
            parameter_space,
            ctx,
        ) for s in stmt.body.args
    ]
    return Expr(:block, initialise_iter_expr, initialise_size_expr, body_exprs...)
end

"""
    Indexed{D,N,T}

Marker struct to indicate that the argument is an object indexed by the batch loop index.

`D` is the dimension that contains the loop index, and `N` is the number of indices that
was passed to the indexing expression.
"""
struct Indexed{D,N,T}
    data::T

    function Indexed{D,N}(data::T) where {D,N,T}
        new{D,N,T}(data)
    end
end

"""
    independent(dists::AbstractArray)

Return a distribution over arrays with the same shape as `dists`, where each entry
is independently drawn from `dists[i]`.

Used by `@batch` for indexed distributions without a specialised `batched` method.
Backends (e.g. the Distributions and ProbabilityMeasures extensions) add methods for
their own distribution types.
"""
function independent end

"""
    iid(dist, dims::Dims)

Return independent draws from `dist` in an array of shape `(size(dist)..., dims...)`.
The distribution is stored once. `@batch` uses this when the distribution does not
depend on the loop index.

```julia
@wren function f(N)
    x ~ iid(Normal(), (N,))   # length N
    W ~ iid(Normal(), (3, N)) # 3 × N
end
```

Inside `@wren`, `@iid y ~ dist` is shorthand for `y ~ iid(dist, size(y))`.

When the lhs is undefined, as is typically the case when setting up priors, use the
`size` keyword to set its shape `@iid mu ~ dist size=k` where k may be either a tuple
or an integer.

The fallback is `independent(Fill(dist, dims))`; backends may add specialised methods.
"""
iid(dist, dims::Dims) = independent(FillArrays.Fill(dist, dims))
iid(dist, n::Integer) = iid(dist, (n,))

# TODO(penelopeysm): I'm unsure about type stability here because of the vararg
# specialisation thing. We might need the fallback on generated functions with enumerated
# arguments
_unindex(@nospecialize(x::Any), i) = x
function _unindex(indexed::Indexed{D,N,<:AbstractVector}, i) where {D,N}
    # Vectors are always safe to un-index, because the single dimension from the vector must
    # have been constructed from the loop range.
    return indexed.data[i]
end
function _unindex(indexed::Indexed{D,N,<:AbstractArray{T,N}}, i) where {D,N,T}
    return selectdim(indexed.data, D, i)
end
function _unindex(indexed::Indexed, i)
    _issue_error("cannot safely unindex type $(typeof(indexed))")
end
function batched(iters::I, ::Val{F}, args...; kwargs...) where {I,F}
    # Arguments are already sliced, so index their positions rather than the loop values.
    return independent([
        F((_unindex(a, i) for a in args)...; kwargs...) for i in LinearIndices(iters)
    ])
end

# This returns `WrenCore.independent([$rhs for $loop_idx in $loop_iters])`.
abstract type BatchFallbackReason end
info(rhs, ::Nothing) = nothing
struct FunctionHasKwargs <: BatchFallbackReason
    fname::Symbol
end
function info(rhs, r::FunctionHasKwargs)
    @info "@batch: batching of `$(rhs)` disabled because the function `$(r.fname)` has keyword arguments that depend on the loop index"
end
struct NotFunctionCall <: BatchFallbackReason end
function info(rhs, ::NotFunctionCall)
    @info "@batch: batching of `$(rhs)` disabled because it is not a function call"
end
struct UnknownExprHead <: BatchFallbackReason
    expr::Any
    is_indexing_arg::Bool
end
function info(rhs, e::UnknownExprHead)
    if e.is_indexing_arg
        @info "@batch: batching of `$(rhs)` disabled because it contains an index `$(e.expr)` that is not the loop index"
    else
        @info "@batch: batching of `$(rhs)` disabled because it contains an argument `$(e.expr)` that is not a simple indexing expression"
    end
end
struct CannotBatch{R<:BatchFallbackReason} <: Exception
    rhs::Expr
    reason::R
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
            error("@batch: encountered unexpected kwarg expression $(kw)")
        end
    end
    return false
end
function _make_indexed(D, N, expr)
    return Expr(:call, Expr(:curly, Indexed, D, N), expr)
end

function batch_expr(expr, loop_idx, loop_iters, loop_iters_size)
    try
        # Check if the expr contains the loop index anywhere.
        return if !occurs_in_expr(expr, loop_idx)
            # If not, it's a constant wrt the loop.
            Expr(:call, WrenCore.iid, expr, loop_iters_size)
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
                    throw(CannotBatch(expr, FunctionHasKwargs(expr.args[1])))
                else
                    # batch the call e.g. f(x[i], y[i])
                    #   --> batched(1:J, Val{f}(), (Indexed{1,1}(x[1:J]), Indexed{1,1}(y[1:J])))
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
                                batched_expr = replace_loop_idx_with_iters(
                                    arg,
                                    loop_idx,
                                    loop_iters,
                                    false,
                                )
                                # TODO(penelopeysm): Should we be using the second return value here to
                                # determine whether or not to view???
                                if occurs_in_expr(arg, loop_idx)
                                    expanded = batched_expr.new_expr
                                    maybe_viewed =
                                        if expanded isa Expr && expanded.head == :ref
                                            Expr(
                                                :macrocall,
                                                Symbol("@view"),
                                                nothing,
                                                expanded,
                                            )
                                        else
                                            expanded
                                        end
                                    _make_indexed(
                                        batched_expr.D,
                                        batched_expr.N,
                                        maybe_viewed,
                                    )
                                else
                                    batched_expr.new_expr
                                end
                            end
                        end,
                        positional_args,
                    )

                    if kwargs_expr === nothing
                        Expr(
                            :call,
                            WrenCore.batched,
                            loop_iters,
                            fname,
                            batched_positional_args...,
                        )
                    else
                        Expr(
                            :call,
                            WrenCore.batched,
                            kwargs_expr,
                            loop_iters,
                            fname,
                            batched_positional_args...,
                        )

                    end
                end
            else
                throw(CannotBatch(expr, NotFunctionCall()))
            end
        end
    catch e
        if e isa CannotBatch
            # Fall back to product of independent distributions
            info(expr, e.reason)
            return Expr(
                :call,
                WrenCore.independent,
                Expr(
                    :comprehension,
                    Expr(:generator, expr, Expr(:(=), loop_idx, loop_iters)),
                ),
            )
        else
            rethrow(e)
        end
    end
end

occurs_in_expr(expr::Symbol, loop_idx::Symbol) = expr == loop_idx
occurs_in_expr(expr::QuoteNode, loop_idx::Symbol) = false
occurs_in_expr(expr::Expr, loop_idx::Symbol) =
    any(arg -> occurs_in_expr(arg, loop_idx), expr.args)
occurs_in_expr(expr, loop_idx) = false

struct BatchedExpr
    "The modified expression. Usually Symbol or Expr, but the type bound is `Any` because it might include literals"
    new_expr::Any
    "Whether or not the expression was batched (i.e., it used to refer to a loop index but now refers to the iteration range)"
    is_batched::Bool
    "The dimension along which the batching occurs"
    D::Union{Nothing,Int}
    "The total number of dimensions"
    N::Union{Nothing,Int}
end

# Needed for unit testing.
Base.:(==)(a::BatchedExpr, b::BatchedExpr) =
    a.new_expr == b.new_expr && a.is_batched == b.is_batched && a.D == b.D && a.N == b.N

"""
    replace_loop_idx_with_iters(expr, loop_idx::Symbol, loop_iters, is_indexing_arg)::BatchedExpr

Replace all occurrences of `loop_idx` in `expr` with `loop_iters` and broadcast where
necessary.

`is_indexing_arg` indicates whether `expr` is an argument to an indexing expression (i.e., `foo[expr]`).
"""
function replace_loop_idx_with_iters(
    expr,
    loop_idx::Symbol,
    loop_iters,
    is_indexing_arg::Bool,
)
    return if expr isa Symbol
        expr == loop_idx ? BatchedExpr(loop_iters, true, nothing, nothing) :
        BatchedExpr(expr, false, nothing, nothing)
    elseif expr isa QuoteNode
        BatchedExpr(expr, false, nothing, nothing)
    elseif expr isa Expr
        if expr.head === :ref
            # expr.args[1] is the thing being indexed into, so we don't need to replace it
            nargs = length(expr.args) - 1
            new_args_and_is_batched = map(
                arg -> replace_loop_idx_with_iters(arg, loop_idx, loop_iters, true),
                expr.args[2:end],
            )
            new_args = map(x -> x.new_expr, new_args_and_is_batched)
            is_batched = map(x -> x.is_batched, new_args_and_is_batched)
            batched_arg_indices = (1:nargs)[is_batched]
            n_batched_args = count(is_batched)

            if n_batched_args == 0
                # e.g. x[const] -> x[const]
                BatchedExpr(
                    Expr(expr.head, expr.args[1], new_args...),
                    false,
                    nothing,
                    nothing,
                )
            elseif n_batched_args > 1
                # e.g. x[i, i] -> getindex.(Ref(x), 1:J, 1:J)
                #
                # The alternative, `x[i, i] for i in 1:J`, creates a closure inside an
                # RGF which makes Enzyme very angry.
                args = map(new_args_and_is_batched) do batched_expr
                    new_expr = batched_expr.new_expr
                    batched_expr.is_batched ? new_expr : Expr(:call, Base.Ref, new_expr)
                end
                array_ref = Expr(:call, Base.Ref, expr.args[1])
                BatchedExpr(
                    Expr(:., Base.getindex, Expr(:tuple, array_ref, args...)),
                    true,
                    nothing,
                    nothing,
                )
            else
                D = only(batched_arg_indices)
                N = length(new_args)
                BatchedExpr(Expr(expr.head, expr.args[1], new_args...), true, D, N)
            end
        else
            if occurs_in_expr(expr, loop_idx)
                throw(CannotBatch(expr, UnknownExprHead(expr, is_indexing_arg)))
            else
                BatchedExpr(expr, false, nothing, nothing)
            end
        end
    else
        # e.g. literals
        BatchedExpr(expr, false, nothing, nothing)
    end
end

vectorise_batch_stmt(lnn::LineNumberNode, idx, iters, size, ax, cond, ls, ps, ctx) = lnn
function vectorise_batch_stmt(::IID, idx, iters, size, ax, cond, ls, ps, ctx)
    return error("@iid is not supported inside @batch")
end
function vectorise_batch_stmt(::TrackedQuantity, idx, iters, size, ax, cond, ls, ps, ctx)
    return error("@track is not supported inside @batch")
end

function vectorise_batch_stmt(
    stmt::Sample,
    loop_idx,
    loop_iters,
    loop_iters_size,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    new_lhs = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters, false).new_expr
    rhs_expr =
        emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space, ctx)
    new_rhs = batch_expr(rhs_expr, loop_idx, loop_iters, loop_iters_size)
    return emit_model_expr(
        Sample(stmt.name, new_lhs, new_rhs, stmt.lhs_top_level_symbol, stmt.lhs_is_reused),
        accs,
        conditioned,
        latent_source,
        parameter_space,
        ctx,
    )
end

function vectorise_batch_stmt(
    stmt::Observe,
    loop_idx,
    loop_iters,
    loop_iters_size,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    new_lhs = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters, false).new_expr
    rhs_expr =
        emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space, ctx)
    new_rhs = batch_expr(rhs_expr, loop_idx, loop_iters, loop_iters_size)
    return emit_model_expr(
        Observe(stmt.name, new_lhs, new_rhs),
        accs,
        conditioned,
        latent_source,
        parameter_space,
        ctx,
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
    ctx,
)
    @gensym value
    value_expr =
        emit_model_expr(stmt.expr, accs, conditioned, latent_source, parameter_space, ctx)
    return quote
        if $(hasacc)($accs, $(Val{LOG_LIKELIHOOD_ACCNAME}()))
            $value = $value_expr
            $accs = $(accumulate_addloglikelihood!!)($accs, $value, $(Val{stmt.name}()))
        end
    end
end

# If the TQ is not `always_evaluated`, `expr` is inside an `is_accumulating_tq`
# branch, which the compiler removes when neither the TQ itself nor any quantity in
# `tqs_triggering_evaluation` is requested. If `tqs_triggering_evaluation` is empty, the
# value is assigned to a gensym so that it cannot shadow a variable of the same name.
function emit_model_expr(
    stmt::TrackedQuantity,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    # If either `stmt.always_evaluated` is true, or one of the TQs in `stmt.tqs_triggering_evaluation`
    # is being collected, then we need to _evaluate_ the value of this TQ. This is separate to whether
    # or not it is collected (in the TQ accumulator), which happens only if `stmt.name` was requested.
    @gensym tq_value
    value_expr =
        emit_model_expr(stmt.expr, accs, conditioned, latent_source, parameter_space, ctx)
    is_being_accumulated = :($(is_accumulating_tq)($accs, $(Val{stmt.name}())))
    return if stmt.always_evaluated
        quote
            # `let` keeps locals of the expression outside of the model body
            $(stmt.name) = let
                $value_expr
            end
            if $is_being_accumulated
                $accs = $(accumulate_tq!!)($accs, $(stmt.name), $(VarName{stmt.name}()))
            end
            $(stmt.name)
        end
    else
        value_name = isempty(stmt.tqs_triggering_evaluation) ? tq_value : stmt.name
        is_needed = foldl(
            stmt.tqs_triggering_evaluation;
            init=is_being_accumulated,
        ) do condition, dependent
            :($condition || $(is_accumulating_tq)($accs, $(Val{dependent}())))
        end
        quote
            if $is_needed
                $value_name = let
                    $value_expr
                end
                if $is_being_accumulated
                    $accs = $(accumulate_tq!!)($accs, $value_name, $(VarName{stmt.name}()))
                end
                $value_name
            else
                nothing
            end
        end
    end
end

function emit_model_expr(
    stmt::Observe,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    @gensym rhs val vn
    rhs_expr =
        emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space, ctx)
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
        $val
    end
end

function emit_model_expr(
    stmt::ReturnStatement,
    accs,
    conditioned,
    latent_source,
    parameter_space,
    ctx,
)
    body = Expr(:block)
    @gensym retval
    if stmt.expr !== nothing
        return_expr = emit_model_expr(
            stmt.expr,
            accs,
            conditioned,
            latent_source,
            parameter_space,
            ctx,
        )
        push!(body.args, :($retval = $return_expr))
    else
        push!(body.args, :($retval = nothing))
    end
    push!(body.args, :(return ($retval, $accs)))
    return body
end

function emit_statements(stmts, accs, conditioned, latent_source, parameter_space, ctx)
    body = Expr(:block)
    for stmt in stmts
        push!(
            body.args,
            emit_model_expr(stmt, accs, conditioned, latent_source, parameter_space, ctx),
        )
    end
    return body
end
