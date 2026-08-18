function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        RawValueAccumulator(),
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
        AccumulatorTuple(RawValueAccumulator()),
    )
    return get_raw_values(accs)
end

Base.rand(model::Model) = rand(default_rng(), model)

conditioned(model::Model) = model.conditioned

function condition(model::Model, values::VarNamedTuple)
    new_conditioned = merge(model.conditioned, values)
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod,
        new_conditioned,
        model.evaluation_fn,
    )
end

Base.:|(model::Model, values::VarNamedTuple) = condition(model, values)
