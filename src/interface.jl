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
    )
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
