"""
    parameters(accs::AccumulatorTuple)::VarNamedTuple

Return the accumulated latent parameter values.
"""
VarNames.parameters(accs::AccumulatorTuple) = getacc(accs, Val(PARAMS_ACCNAME)).values

"""Return the vectorised parameter values."""
vectorised_params(accs::AccumulatorTuple) =
    getacc(accs, Val(VECTORISED_PARAMS_ACCNAME)).values

"""
    logprior(accs::AccumulatorTuple)::Number

Return the accumulated log-prior.

This number is always in original coordinates, even if the model was evaluated in
transformed coordinates. If you want the log-prior density in transformed coordinates, you
can compute it as `logprior(accs) - logjac(accs)`.
"""
@inline logprior(accs::AccumulatorTuple) = getacc(accs, Val(LOG_PRIOR_ACCNAME)).logp

"""
    StatsAPI.loglikelihood(accs::AccumulatorTuple)::Number

Return the accumulated log-likelihood.
"""
@inline StatsAPI.loglikelihood(accs::AccumulatorTuple) =
    getacc(accs, Val(LOG_LIKELIHOOD_ACCNAME)).logp

"""
    logjac(accs::AccumulatorTuple)::Number

Return the accumulated log-absolute determinant of the Jacobian of the transformation **from
original to the space that the model was evaluated in**. If the model is evaluated in original
space, this will be zero. However, if the model was evaluated in transformed space, then this
is `J_f(x)` where `f` is the _forward_ transformation into the evaluation space.

Consequently, the log-joint density in transformed space is given by

    logjoint_transformed(accs) = logjoint(accs) - logjac(accs)

(note the minus sign).
"""
@inline logjac(accs::AccumulatorTuple) = getacc(accs, Val(LOG_JACOBIAN_ACCNAME)).logjac

"""
    logjoint(accs::AccumulatorTuple)::Number

Return the log joint density in original coordinates, i.e. the sum of the log-prior and
log-likelihood.
"""
@inline logjoint(accs::AccumulatorTuple) = logprior(accs) + loglikelihood(accs)

"""
    logjoint_transformed(accs::AccumulatorTuple)::Number

Return the log joint density in transformed coordinates, i.e.

    logjoint_transformed(accs) = logjoint(accs) - logjac(accs).
"""
@inline logjoint_transformed(accs::AccumulatorTuple) = logjoint(accs) - logjac(accs)
