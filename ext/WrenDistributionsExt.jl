module WrenDistributionsExt

using Wren: Wren
using Random: Random
using Plaice: Plaice
using FillArrays: Fill
using Distributions: Distribution, product_distribution, Normal

Wren._product(dists::AbstractArray{<:Distribution}) = product_distribution(dists)

Wren.batched(iters, dist::Distribution) = product_distribution(Fill(dist, length(iters)))

struct IsoNormal{M<:AbstractArray}
    mu::M
end
Plaice.from_unconstrained_vec(d::IsoNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::IsoNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .+ d.mu
function Wren.logdensityof(d::IsoNormal, x)
    dist = mapreduce((a, b) -> abs2(a - b), +, d.mu, x)
    return -0.5 * dist - 0.5 * length(d.mu) * log(2π)
end
function Wren.batched(iters, ::Val{Normal}, tup::T) where {T<:Tuple{<:Wren.Indexed}}
    # e.g. batched(1:10, Val(Normal), (Indexed(x[1:10]),)
    return IsoNormal(tup[1].data)
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
    dist = sum(xi -> abs2((xi - d.mu) * inv_sigma), x)
    N = length(x)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log(2π)
end
Wren.batched(iters, n::Normal) = IIDNormal(size(iters), n.μ, n.σ)

end # module WrenDistributionsExt
