"""Return the raw, untransformed model values."""
get_raw_values(accs::AccumulatorTuple) = getacc(accs, Val(RAW_VALUE_ACCNAME)).values

"""Return the vectorised parameter values."""
get_vector_values(accs::AccumulatorTuple) = getacc(accs, Val(VECTOR_VALUE_ACCNAME)).values

"""Return the accumulated log prior."""
getlogprior(accs::AccumulatorTuple) = getacc(accs, Val(LOG_PRIOR_ACCNAME)).logp

"""Return the accumulated log likelihood."""
getloglikelihood(accs::AccumulatorTuple) = getacc(accs, Val(LOG_LIKELIHOOD_ACCNAME)).logp

"""Return the cumulative forward log-Jacobian."""
getlogjac(accs::AccumulatorTuple) = getacc(accs, Val(LOG_JACOBIAN_ACCNAME)).logjac

"""Return the log joint density in the distribution's original coordinates."""
getlogjoint(accs::AccumulatorTuple) = getlogprior(accs) + getloglikelihood(accs)

"""Return the log joint density in transformed coordinates."""
getlogjoint_transformed(accs::AccumulatorTuple) = getlogjoint(accs) - getlogjac(accs)
