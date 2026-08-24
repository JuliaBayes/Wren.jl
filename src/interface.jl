function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        ParamsAccumulator(),
    )
end

"""
    evaluate(model, latent_source, parameter_space, accs)

Evaluate `model` in three stages for every latent variable:

1. `latent_source` supplies a [`TransformedValue`](@ref).
2. `parameter_space` determines the model-space value, target representation, and
   forward log-Jacobian.
3. `accs` accumulate the resulting information.

All three components are required. Higher-level interfaces may choose defaults, but this
method keeps the evaluation pipeline explicit.
"""
@inline function evaluate(
    model::Model,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple,
)
    return model.evaluation_fn(
        latent_source,
        parameter_space,
        accs,
        model.args,
        model.conditioned,
    )
end

function Base.rand(rng::AbstractRNG, model::Model)
    _, accs = evaluate(
        model,
        FromPrior(rng),
        OriginalSpace(),
        AccumulatorTuple(ParamsAccumulator()),
    )
    return VarNames.parameters(accs)
end
Base.rand(model::Model) = rand(default_rng(), model)

"""
    rand_with_logp([rng::Random.AbstractRNG,] model::Model)::AccumulatorTuple

Draw a sample from the prior distribution of `model` and return the params, log-prior,
and log-likelihood.

To extract the contents of the `AccumulatorTuple`, you can use [`parameters`](@ref),
[`logprior`](@ref), [`loglikelihood`](@ref), and [`logjoint`](@ref).
"""
function rand_with_logp(rng::AbstractRNG, model::Model)
    _, accs = evaluate(
        model,
        FromPrior(rng),
        OriginalSpace(),
        AccumulatorTuple(
            ParamsAccumulator(),
            LogPriorAccumulator(),
            LogLikelihoodAccumulator(),
        ),
    )
    return accs
end
rand_with_logp(model::Model) = rand_with_logp(default_rng(), model)

"""
    conditioned(model::Model)::VarNamedTuple

Return the current set of values that `model` is conditioned on.
"""
conditioned(model::Model) = model.conditioned

"""
    _replace_conditioned(model::Model, conditioned::VarNamedTuple)::Model

Return a copy of `model` with its conditioned values replaced by
`conditioned`. Internal helper for [`condition`](@ref) and
[`decondition`](@ref).
"""
function _replace_conditioned(model::Model, conditioned::VarNamedTuple)
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned,
        evaluation_fn=model.evaluation_fn,
    )
end

"""
    _without(vnt::VarNamedTuple, vns)::VarNamedTuple

Return a copy of `vnt` without the entries subsumed by any of `vns`.
The complement of `VarNames.subset`.
"""
function _without(vnt::VarNamedTuple, vns)
    isempty(vnt) && return vnt
    return mapfoldl(
        identity,
        function (acc, pair)
            name, value = pair
            if any(vn -> VarNames.subsumes(vn, name), vns)
                acc
            else
                VarNames.templated_setindex!!(
                    acc,
                    value,
                    name,
                    vnt.data[VarNames.getsym(name)],
                )
            end
        end,
        vnt;
        init=VarNamedTuple(),
    )
end

"""
    condition(model::Model, values::VarNamedTuple)::Model

Condition `model` on the values in `values`, returning a new model. The original model is
not modified.

If a top-level key of `values` names a submodel site, the corresponding subtree of values
is pushed down into that submodel: `model | @vnt begin x.obs := data end` conditions `obs`
inside the submodel at site `x`, by conditioning the submodel and rewriting it back in.
Conditioning the same site again merges values. See the manual page on rewriting models
for details and requirements.
"""
function condition(model::Model, values::VarNamedTuple)
    m = model
    pushed = VarName[]
    for sym in keys(values.data)
        inner = _submodel_site_value(m, sym)
        inner === nothing && continue
        vn = VarName{sym}()
        subtree = values[vn]
        subtree isa VarNamedTuple || error(
            "condition: the site `$sym` is a submodel, so it must be " *
            "conditioned on values for its variables (e.g. `@vnt begin " *
            "$sym.var := value end`), not on a raw value of type " *
            "$(typeof(subtree)). To replace the submodel wholesale, use " *
            "`rewrite(model, @vn($sym) => value)` instead.",
        )
        m = rewrite(m, vn => condition(inner, subtree))
        push!(pushed, vn)
    end
    remaining = isempty(pushed) ? values : _without(values, pushed)
    return _replace_conditioned(m, merge(conditioned(m), remaining))
end

"""
    decondition(model::Model, vns::VarName...)::Model

Remove the conditioning on the given variables, returning a new model. The original model
is not modified. A `vn` removes all conditioned values that it subsumes, so e.g.
`decondition(model, @vn(x))` removes `x[1]` and `x.a` as well as `x` itself, including
conditioning that [`condition`](@ref) pushed down into a submodel site.
"""
function decondition(model::Model, vns::VarName...)
    m = model
    for vn in vns
        sym = VarNames.getsym(vn)
        inner = _submodel_site_value(m, sym)
        inner isa Model || continue
        optic = VarNames.getoptic(vn)
        new_inner = if optic isa VarNames.Iden
            _decondition_all(inner)
        else
            decondition(inner, VarNames.unprefix(vn, VarName{sym}()))
        end
        if new_inner !== inner
            m = rewrite(m, VarName{sym}() => new_inner)
        end
    end
    return _replace_conditioned(m, _without(conditioned(m), vns))
end

"""
    _decondition_all(model::Model)::Model

Remove all conditioning from `model`, including conditioning pushed down into submodel
sites (recursively). Returns `model` itself if there is nothing to remove.
"""
function _decondition_all(model::Model)
    m = isempty(conditioned(model)) ? model : _replace_conditioned(model, VarNamedTuple())
    syms = Symbol[]
    for stmt in m.stmts
        stmt isa LineNumberNode || _plain_sample_syms!(syms, stmt)
    end
    for sym in syms
        inner = _submodel_site_value(m, sym)
        inner isa Model || continue
        new_inner = _decondition_all(inner)
        if new_inner !== inner
            m = rewrite(m, VarName{sym}() => new_inner)
        end
    end
    return m
end

_plain_sample_syms!(syms::Vector{Symbol}, stmt) = syms
function _plain_sample_syms!(syms::Vector{Symbol}, stmt::Sample)
    stmt.lhs isa Symbol && push!(syms, stmt.lhs)
    return syms
end
function _plain_sample_syms!(syms::Vector{Symbol}, stmt::WrenExpr)
    for arg in stmt.args
        _plain_sample_syms!(syms, arg)
    end
    return syms
end

"""
    Base.:|(model::Model, values::VarNamedTuple)::Model

`model | values` is an alias for [`condition(model, values)`](@ref).
"""
Base.:|(model::Model, values::VarNamedTuple) = condition(model, values)

"""
    logjoint(model::Model, params::VarNamedTuple)

Return the log-joint probability of `model` evaluated at `params` in original space.
"""
function logjoint(model::Model, params::VarNamedTuple)
    _, accs = evaluate(
        model,
        FromParams(params),
        OriginalSpace(),
        AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator()),
    )
    return logjoint(accs)
end
