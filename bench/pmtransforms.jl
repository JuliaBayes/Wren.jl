module PMTransforms

using DensityInterface: DensityInterface
using ProbabilityMeasures: ProbabilityMeasures as PM
using Random: AbstractRNG, randn
using VectorBijectors: VectorBijectors
using Wren: Wren

export Categorical, IsoNormal, MvCategorical

# Bijector boilerplate for PM.Normal().

struct Only end
Only(x::AbstractVector{<:Number}) = x[]
function VectorBijectors.with_logabsdet_jacobian(::Only, x::AbstractVector{<:Number})
    return x[], zero(eltype(x))
end
VectorBijectors.inverse(::Only) = Vec()

struct Vec end
Vec(r) = [r]
VectorBijectors.with_logabsdet_jacobian(::Vec, v) = ([v], zero(typeof(v)))
VectorBijectors.inverse(::Vec) = Only()

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

function VectorBijectors.with_logabsdet_jacobian(
    transform::FromIsoNormalVec,
    x::AbstractVector{<:Number},
)
    return transform(x), zero(eltype(x))
end
function VectorBijectors.with_logabsdet_jacobian(
    transform::ToIsoNormalVec,
    x::AbstractArray{<:Number},
)
    return transform(x), zero(eltype(x))
end

VectorBijectors.inverse(transform::FromIsoNormalVec) = ToIsoNormalVec(transform.dims)
VectorBijectors.inverse(transform::ToIsoNormalVec) = FromIsoNormalVec(transform.dims)
VectorBijectors.from_linked_vec(d::IsoNormal) = FromIsoNormalVec(d.dims)
VectorBijectors.from_vec(d::IsoNormal) = FromIsoNormalVec(d.dims)

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

function DensityInterface.logdensityof(d::MvCategorical, categories::AbstractVector)
    probabilities = d.probabilities
    linear_indices = categories .+ (eachindex(categories) .- 1) .* size(probabilities, 1)
    return sum(log, probabilities[linear_indices])
end

end # module PMTransforms
