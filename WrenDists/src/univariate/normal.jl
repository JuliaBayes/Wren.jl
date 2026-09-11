struct Normal{M<:Number,S<:Number} <: UnconstrainedUnivariateDist
    μ::M
    σ::S
end

Normal() = Normal(0.0, 1.0)

@inline zval(d::Normal, x::Number) = (x - d.μ) / d.σ
@inline xval(d::Normal, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Normal, x::Number)
    z = zval(d, x)
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return -(z^2 + log2π) / 2 - logt(σ)
end

# Apply `μ` and `σ` after drawing noise so automatic differentiation can follow them.
@inline function Base.rand(rng::AbstractRNG, d::Normal)
    return xval(d, randn(rng, noisetype(d)))
end

# --- Batched Normal distributions ---

# Compute the shared normalising constant once.
function _iid_logdensityof(dist::Normal, x::AbstractArray)
    μ = dist.μ
    _, _, σ = promote(zero(eltype(x)), μ, dist.σ)
    inv_σ = inv(float(σ))
    z2 = sum(xi -> abs2((xi - μ) * inv_σ), x; init=zero(inv_σ))
    lognorm = logdensityof(dist, oftype(z2, μ))
    return length(x) * lognorm - z2 / 2
end

struct DiagNormal{M<:AbstractArray,S<:AbstractArray}
    mu::M
    sigma::S
end
Plaice.from_unconstrained_vec(d::DiagNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::DiagNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::DiagNormal) =
    randn(rng, size(d.mu)) .* d.sigma .+ d.mu
@inline function DensityInterface.logdensityof(d::DiagNormal, x)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x, d.mu, d.sigma)
        z = (x[i] - d.mu[i]) / d.sigma[i]
        s += z * z + 2 * log(d.sigma[i])
    end
    return -0.5 * s - 0.5 * length(d.mu) * log(2π)
end
struct IsoNormal{M<:AbstractArray}
    mu::M
end
Plaice.from_unconstrained_vec(d::IsoNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::IsoNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .+ d.mu
function DensityInterface.logdensityof(d::IsoNormal, x)
    dist = sum(abs2, x .- d.mu)
    return -0.5 * dist - 0.5 * length(d.mu) * log(2π)
end
struct IIDNormal{T<:Tuple,M<:Number,S<:Number}
    size::T
    mu::M
    sigma::S
end
Base.rand(rng::Random.AbstractRNG, d::IIDNormal) = randn(rng, d.size) .* d.sigma .+ d.mu
Plaice.from_unconstrained_vec(d::IIDNormal) = Plaice.Reshape(d.size)
Plaice.to_unconstrained_vec(d::IIDNormal) = Plaice.Vec(d.size)
function DensityInterface.logdensityof(d::IIDNormal, x)
    inv_sigma = inv(d.sigma)
    # dist = sum(xi -> abs2((xi - d.mu) * inv_sigma), x)
    # ^ This is faster, but creates lots of allocs
    dist = sum(abs2, (x .- d.mu) .* inv_sigma)
    N = length(x)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log(2π)
end
