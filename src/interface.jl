function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        RawValueAccumulator(),
    )
end

@inline function evaluate(
    model::Model,
    latent_source::AbstractLatentSource,
    accs::AccumulatorTuple,
)
    return model.evaluation_fn(latent_source, accs, model.args, model.conditioned)
end

function Base.rand(rng::AbstractRNG, model::Model)
    _, accs = evaluate(model, FromPrior(rng), AccumulatorTuple(RawValueAccumulator()))
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
