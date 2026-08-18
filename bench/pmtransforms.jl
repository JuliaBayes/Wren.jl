using ChangesOfVariables: ChangesOfVariables
using InverseFunctions: InverseFunctions
using ProbabilityMeasures: ProbabilityMeasures as PM
using Wren: Wren

# Bijector boilerplate for PM.Normal().

struct Only end
Only(x::AbstractVector{<:Number}) = x[]
ChangesOfVariables.with_logabsdet_jacobian(::Only, x::AbstractVector{<:Number}) = (x[], zero(eltype(x)))
InverseFunctions.inverse(::Only) = Vec()
struct Vec end
Vec(r) = [r]
ChangesOfVariables.with_logabsdet_jacobian(::Vec, v) = ([v], zero(eltype(v)))
InverseFunctions.inverse(::Vec) = Only()
Wren.from_unconstrained_vec(::PM.Normal) = Only()
Wren.from_vec(::PM.Normal) = Only()
