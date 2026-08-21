module WrenDistributionsExt

using Wren: Wren
using Random: Random
using FillArrays: Fill
using Distributions: Distribution, product_distribution, Normal

Wren._product(dists::AbstractArray{<:Distribution}) = product_distribution(dists)

Wren.batched(iters, dist::Distribution) = product_distribution(Fill(dist, length(iters)))

struct IsoNormal{M<:AbstractArray}
    mu::M
end
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .+ d.mu
function Wren.logdensityof(d::IsoNormal, x)
    dist = sum(abs2, d.mu .- x)
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
function Wren.logdensityof(d::IIDNormal, x)
    dist = sum(abs2, (x .- d.mu) ./ d.sigma)
    N = length(x)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log(2π)
end
Wren.batched(iters, n::Normal) = IIDNormal(size(iters), n.μ, n.σ)

end # module WrenDistributionsExt
