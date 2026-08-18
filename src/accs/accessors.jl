"""Return the raw, untransformed model values."""
get_raw_values(accs::AccumulatorTuple) = getacc(accs, Val(RAW_VALUE_ACCNAME)).values

"""Return the accumulated log prior."""
getlogprior(accs::AccumulatorTuple) = getacc(accs, Val(:LogPrior)).logp

"""Return the accumulated log likelihood."""
getloglikelihood(accs::AccumulatorTuple) = getacc(accs, Val(:LogLikelihood)).logp

"""Return the cumulative forward log-Jacobian."""
getlogjac(accs::AccumulatorTuple) = getacc(accs, Val(:LogJacobian)).logjac

"""Return the log joint density in the distribution's original coordinates."""
getlogjoint(accs::AccumulatorTuple) = getlogprior(accs) + getloglikelihood(accs)

"""Return the log joint density in transformed coordinates."""
getlogjoint_transformed(accs::AccumulatorTuple) = getlogjoint(accs) - getlogjac(accs)
