function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        ParamsAccumulator(),
    )
end

"""
    evaluate(model, latent_source, parameter_space, accs, ctx[, conditioned])

Evaluate `model` in three stages for every latent variable:

1. `latent_source` supplies a [`TransformedValue`](@ref).
2. `parameter_space` determines the model-space value, target representation, and
   forward log-Jacobian.
3. `accs` accumulate the resulting information.

All three components are required. Higher-level interfaces may choose defaults, but this
method keeps the evaluation pipeline explicit.

`ctx` is an [`EvalContext`](@ref), which says whether to make the copies of model arguments
that Enzyme needs. Unless the model is being differentiated with Enzyme, this should be
`EvalContext{false}()`.

`conditioned` defaults to `conditioned(model)`. A parent model passes other values when it
evaluates a submodel.
"""
@inline function evaluate(
    model::Model,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple,
    ctx::EvalContext,
    conditioned::VarNamedTuple=model.conditioned,
)
    return model.evaluation_fn(
        latent_source,
        parameter_space,
        accs,
        model.args,
        conditioned,
        model.enzyme_copy_argnames,
        ctx,
    )
end

function Base.rand(rng::AbstractRNG, model::Model)
    _, accs = evaluate(
        model,
        FromPrior(rng),
        OriginalSpace(),
        AccumulatorTuple(ParamsAccumulator()),
        EvalContext{false}(),
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
        EvalContext{false}(),
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
    condition(model::Model, values::VarNamedTuple)::Model

Condition `model` on the values in `values`, returning a new model. The original model is
not modified.
"""
function condition(model::Model, values::VarNamedTuple)
    new_conditioned = merge(conditioned(model), values)
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned=new_conditioned,
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end

"""
    Base.:|(model::Model, values::VarNamedTuple)::Model

`model | values` is an alias for [`condition(model, values)`](@ref).
"""
Base.:|(model::Model, values::VarNamedTuple) = condition(model, values)

"""
    decondition(model::Model, vns::VarName...)::Model

Remove `vns` from the set of variables that `model` is conditioned on, returning a new
model. A variable is removed if it is subsumed by any of `vns`, so `decondition(model,
@vn(x))` also removes `x[1]` and `x.a`. With no `vns`, every conditioned variable is
removed.

Variables observed with `@observe` are part of the model's structure rather than its
conditioning, and are unaffected.
"""
function decondition(model::Model, vns::VarName...)
    old = conditioned(model)
    new_conditioned = if isempty(vns)
        VarNamedTuple()
    else
        kept = [vn for (vn, _) in pairs(old) if !any(v -> VarNames.subsumes(v, vn), vns)]
        VarNames.subset(old, kept)
    end
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned=new_conditioned,
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end

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
        EvalContext{false}(),
    )
    return logjoint(accs)
end
