module WrenCoreDistributionsExt

using WrenCore: WrenCore
using Random: Random
using Plaice: Plaice
using FillArrays: Fill
using Distributions: Distribution, product_distribution, Normal

WrenCore.independent(dists::AbstractArray{<:Distribution}) = product_distribution(dists)

WrenCore.batched(iters, dist::Distribution; _kwargs...) =
    product_distribution(Fill(dist, length(iters)))

struct DiagNormal{M<:AbstractArray,S<:AbstractArray}
    mu::M
    sigma::S
end
Plaice.from_unconstrained_vec(d::DiagNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::DiagNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::DiagNormal) =
    randn(rng, size(d.mu)) .* d.sigma .+ d.mu
@inline function WrenCore.logdensityof(d::DiagNormal, x)
    z = (x .- d.mu) ./ d.sigma
    return -0.5 * sum(abs2, z) - sum(log, d.sigma) - 0.5 * length(d.mu) * log(2π)
end
function WrenCore.batched(
    iters,
    ::Val{Normal},
    indexed_mu::WrenCore.Indexed,
    indexed_sigma::WrenCore.Indexed;
    _kwargs...,
)
    # e.g. batched(1:10, Val(Normal), Indexed(x[1:10]), Indexed(y[1:10]))
    return DiagNormal(indexed_mu.data, indexed_sigma.data)
end

struct IsoNormal{M<:AbstractArray,S<:Number}
    mu::M
    sigma::S
end
IsoNormal(mu::AbstractArray) = IsoNormal(mu, one(eltype(mu)))
Plaice.from_unconstrained_vec(d::IsoNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::IsoNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .* d.sigma .+ d.mu
function WrenCore.logdensityof(d::IsoNormal, x)
    inv_sigma = inv(d.sigma)
    dist = sum(abs2, (x .- d.mu) .* inv_sigma)
    N = length(d.mu)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log(2π)
end
function WrenCore.batched(iters, ::Val{Normal}, indexed_mu::WrenCore.Indexed; _kwargs...)
    # e.g. batched(1:10, Val(Normal), Indexed(x[1:10]))
    return IsoNormal(indexed_mu.data)
end

struct IIDNormal{T<:Tuple,M<:Number,S<:Number}
    size::T
    mu::M
    sigma::S
end
Base.rand(rng::Random.AbstractRNG, d::IIDNormal) = randn(rng, d.size) .* d.sigma .+ d.mu
Plaice.from_unconstrained_vec(d::IIDNormal) = Plaice.Reshape(d.size)
Plaice.to_unconstrained_vec(d::IIDNormal) = Plaice.Vec(d.size)
function WrenCore.logdensityof(d::IIDNormal, x)
    inv_sigma = inv(d.sigma)
    # dist = sum(xi -> abs2((xi - d.mu) * inv_sigma), x)
    # ^ This is faster, but creates lots of allocs
    dist = sum(abs2, (x .- d.mu) .* inv_sigma)
    N = length(x)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log(2π)
end
WrenCore.batched(iters, n::Normal; _kwargs...) = IIDNormal(size(iters), n.μ, n.σ)

end # module WrenCoreDistributionsExt
