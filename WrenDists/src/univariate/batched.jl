# --- Batched univariate distributions ---
#
# A batched univariate distribution stores one array per parameter of the corresponding
# scalar distribution, and represents an array of independent draws with matching shape.
# Constant parameters are stored as `Fill` arrays. `WrenCore.batched` returns these for
# `@batch` loops whose distribution arguments are indexed by the loop variable.

#
# Log densities are written as loops with a scalar accumulator, which Enzyme differentiates
# several times faster than a broadcast (see bench/batch). Loop bodies are written out
# directly rather than calling the scalar `logdensityof`, since its support checks slow
# Enzyme down.

abstract type BatchedUnivariate <: MultivariateDist end
abstract type DiscreteBatchedUnivariate <: DiscreteMultivariateDist end
const AnyBatchedUnivariate = Union{BatchedUnivariate,DiscreteBatchedUnivariate}

# --- Plaice interface ---

Plaice.from_vec(d::AnyBatchedUnivariate) = Plaice.Reshape(size(d))
Plaice.to_vec(d::AnyBatchedUnivariate) = Plaice.Vec(size(d))
Plaice.vec_length(d::AnyBatchedUnivariate) = prod(size(d))
Plaice.optic_vec(d::AnyBatchedUnivariate) = _index_optics(d)

# Discrete batched distributions get the identity unconstrained transform from
# `DiscreteMultivariateDist`. Continuous ones apply the scalar bijector elementwise.
Plaice.scalar_to_scalar_bijector(::BatchedUnivariate) = Plaice.TypedIdentity()
function Plaice.from_unconstrained_vec(d::BatchedUnivariate)
    return ElementwiseInvLink(Plaice.inverse(Plaice.scalar_to_scalar_bijector(d)), size(d))
end
function Plaice.to_unconstrained_vec(d::BatchedUnivariate)
    return ElementwiseLink(Plaice.scalar_to_scalar_bijector(d), size(d))
end
Plaice.unconstrained_vec_length(d::BatchedUnivariate) = prod(size(d))
Plaice.unconstrained_optic_vec(d::BatchedUnivariate) = _index_optics(d)
