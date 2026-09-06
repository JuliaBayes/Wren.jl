module WrenProbabilityMeasuresExt

using Wren: Wren
using Plaice: Plaice
using Random: Random
using ProbabilityMeasures: ProbabilityMeasures as PM

struct DiagNormal{M<:AbstractArray,S<:AbstractArray}
    mu::M
    sigma::S
end
Plaice.from_unconstrained_vec(d::DiagNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::DiagNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::DiagNormal) =
    randn(rng, size(d.mu)) .* d.sigma .+ d.mu
@inline function Wren.logdensityof(d::DiagNormal, x)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x, d.mu, d.sigma)
        z = (x[i] - d.mu[i]) / d.sigma[i]
        s += z * z + 2 * log(d.sigma[i])
    end
    return -0.5 * s - 0.5 * length(d.mu) * log(2π)
end
function Wren.batched(
    iters,
    ::Val{PM.Normal},
    indexed_mu::Wren.Indexed,
    indexed_sigma::Wren.Indexed;
    _kwargs...,
)
    # e.g. batched(1:10, Val(PM.Normal), Indexed(x[1:10]), Indexed(y[1:10]))
    return DiagNormal(indexed_mu.data, indexed_sigma.data)
end

struct IsoNormal{M<:AbstractArray}
    mu::M
end
Plaice.from_unconstrained_vec(d::IsoNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::IsoNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .+ d.mu
function Wren.logdensityof(d::IsoNormal, x)
    dist = sum(abs2, x .- d.mu)
    return -0.5 * dist - 0.5 * length(d.mu) * log(2π)
end
function Wren.batched(iters, ::Val{PM.Normal}, indexed_mu::Wren.Indexed; _kwargs...)
    # e.g. batched(1:10, Val(PM.Normal), Indexed(x[1:10]))
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
function Wren.logdensityof(d::IIDNormal, x)
    inv_sigma = inv(d.sigma)
    # dist = sum(xi -> abs2((xi - d.mu) * inv_sigma), x)
    # ^ This is faster, but creates lots of allocs
    dist = sum(abs2, (x .- d.mu) .* inv_sigma)
    N = length(x)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log(2π)
end
Wren.batched(iters, n::PM.Normal; _kwargs...) = IIDNormal(size(iters), n.μ, n.σ)

end # module WrenProbabilityMeasuresExt
