module PMTransforms

using ChangesOfVariables: ChangesOfVariables
using DensityInterface: DensityInterface
using InverseFunctions: InverseFunctions
using ProbabilityMeasures: ProbabilityMeasures as PM
using Random: AbstractRNG, randn
using Wren: Wren

export Categorical, IsoNormal, MvCategorical

# Bijector boilerplate for PM.Normal().

struct Only end
Only(x::AbstractVector{<:Number}) = x[]
function ChangesOfVariables.with_logabsdet_jacobian(
    ::Only, x::AbstractVector{<:Number}
)
    return x[], zero(eltype(x))
end
InverseFunctions.inverse(::Only) = Vec()

struct Vec end
Vec(r) = [r]
ChangesOfVariables.with_logabsdet_jacobian(::Vec, v) = ([v], zero(typeof(v)))
InverseFunctions.inverse(::Vec) = Only()

Wren.from_unconstrained_vec(::PM.Normal) = Only()
Wren.from_vec(::PM.Normal) = Only()

"An array of independent standard-normal variates with statically known shape."
struct IsoNormal{N}
    dims::NTuple{N,Int}
end
IsoNormal(dims::Integer...) = IsoNormal(map(Int, dims))

Base.rand(rng::AbstractRNG, d::IsoNormal) = randn(rng, d.dims...)

function DensityInterface.logdensityof(::IsoNormal, x::AbstractArray{<:Number})
    return -(sum(abs2, x) + length(x) * log(2π)) / 2
end

struct FromIsoNormalVec{N}
    dims::NTuple{N,Int}
end
(transform::FromIsoNormalVec)(x::AbstractVector{<:Number}) = reshape(x, transform.dims)

struct ToIsoNormalVec{N}
    dims::NTuple{N,Int}
end
(::ToIsoNormalVec)(x::AbstractArray{<:Number}) = vec(x)

function ChangesOfVariables.with_logabsdet_jacobian(
    transform::FromIsoNormalVec, x::AbstractVector{<:Number}
)
    return transform(x), zero(eltype(x))
end
function ChangesOfVariables.with_logabsdet_jacobian(
    transform::ToIsoNormalVec, x::AbstractArray{<:Number}
)
    return transform(x), zero(eltype(x))
end

InverseFunctions.inverse(transform::FromIsoNormalVec) = ToIsoNormalVec(transform.dims)
InverseFunctions.inverse(transform::ToIsoNormalVec) = FromIsoNormalVec(transform.dims)
Wren.from_unconstrained_vec(d::IsoNormal) = FromIsoNormalVec(d.dims)
Wren.from_vec(d::IsoNormal) = FromIsoNormalVec(d.dims)

"A categorical measure parameterized by a probability vector."
struct Categorical{P<:AbstractVector}
    probabilities::P
end

function DensityInterface.logdensityof(d::Categorical, category::Number)
    return log(d.probabilities[category])
end

"Independent categorical measures stored column-wise in a probability matrix."
struct MvCategorical{P<:AbstractMatrix}
    probabilities::P
end

function DensityInterface.logdensityof(
    d::MvCategorical, categories::AbstractVector
)
    probabilities = d.probabilities
    linear_indices =
        categories .+ (eachindex(categories) .- 1) .* size(probabilities, 1)
    return sum(log, probabilities[linear_indices])
end

end # module PMTransforms
