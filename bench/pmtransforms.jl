module PMTransforms

using DensityInterface: DensityInterface
using ProbabilityMeasures: ProbabilityMeasures as PM
using Random: AbstractRNG, randn
using Plaice: Plaice
using WrenCore: WrenCore

export Categorical, IsoNormal, MvCategorical

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

function Plaice.with_logabsdet_jacobian(
    transform::FromIsoNormalVec,
    x::AbstractVector{<:Number},
)
    return transform(x), zero(eltype(x))
end
function Plaice.with_logabsdet_jacobian(
    transform::ToIsoNormalVec,
    x::AbstractArray{<:Number},
)
    return transform(x), zero(eltype(x))
end
Plaice.inverse(transform::FromIsoNormalVec) = ToIsoNormalVec(transform.dims)
Plaice.inverse(transform::ToIsoNormalVec) = FromIsoNormalVec(transform.dims)
Plaice.from_unconstrained_vec(d::IsoNormal) = FromIsoNormalVec(d.dims)
Plaice.from_vec(d::IsoNormal) = FromIsoNormalVec(d.dims)

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
