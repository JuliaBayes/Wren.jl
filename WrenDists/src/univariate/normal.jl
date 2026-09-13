"""
    Normal(μ, σ)
    Normal(μ)
    Normal()

Normal distribution with mean `μ` (default 0.0) and standard deviation `σ` (default 1.0).
"""
struct Normal{M<:Number,S<:Number} <: UnconstrainedUnivariateDist
    μ::M
    σ::S
end

Normal(μ) = Normal(μ, one(μ))
Normal() = Normal(0.0, 1.0)

@inline zval(d::Normal, x::Number) = (x - d.μ) / d.σ
@inline xval(d::Normal, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Normal, x::Number)
    z = zval(d, x)
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return -(z^2 + log2π) / 2 - log(σ)
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

abstract type AbstractBatchedNormal <: MultivariateDist end

struct DiagNormal{M<:AbstractArray,S<:AbstractArray} <: AbstractBatchedNormal
    mu::M
    sigma::S
end
Base.size(d::DiagNormal) = size(d.mu)
Base.rand(rng::Random.AbstractRNG, d::DiagNormal) =
    randn(rng, size(d.mu)) .* d.sigma .+ d.mu
@inline function DensityInterface.logdensityof(d::DiagNormal, x)
    z = (x .- d.mu) ./ d.sigma
    return -0.5 * sum(abs2, z) - sum(log, d.sigma) - 0.5 * length(d.mu) * log2π
end

# TODO(penelopeysm): Some preliminary benchmarks on this show that there's no meaningful
# difference vs. DiagNormal with a FillArray for `sigma`. We should verify this more
# scientifically, and if true, remove this type to make life simpler.
struct IsoNormal{M<:AbstractArray,S<:Number} <: AbstractBatchedNormal
    mu::M
    sigma::S
end
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .* d.sigma .+ d.mu
Base.size(d::IsoNormal) = size(d.mu)
function DensityInterface.logdensityof(d::IsoNormal, x)
    inv_sigma = inv(d.sigma)
    dist = sum(abs2, (x .- d.mu) .* inv_sigma)
    N = length(d.mu)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log2π
end

struct IIDNormal{T<:Tuple,M<:Number,S<:Number} <: AbstractBatchedNormal
    size::T
    mu::M
    sigma::S
end
Base.rand(rng::Random.AbstractRNG, d::IIDNormal) = randn(rng, d.size) .* d.sigma .+ d.mu
Base.size(d::IIDNormal) = d.size
function DensityInterface.logdensityof(d::IIDNormal, x)
    inv_sigma = inv(d.sigma)
    # dist = sum(xi -> abs2((xi - d.mu) * inv_sigma), x)
    # ^ This is faster, but creates lots of allocs
    dist = sum(abs2, (x .- d.mu) .* inv_sigma)
    N = length(x)
    return -0.5 * dist - N * log(d.sigma) - 0.5 * N * log(2π)
end


Plaice.from_unconstrained_vec(d::AbstractBatchedNormal) = Plaice.Reshape(size(d))
Plaice.to_unconstrained_vec(d::AbstractBatchedNormal) = Plaice.Vec(size(d))
Plaice.unconstrained_vec_length(d::AbstractBatchedNormal) = prod(size(d))
Plaice.vec_length(d::AbstractBatchedNormal) = prod(size(d))
Plaice.is_continuous(::AbstractBatchedNormal) = true
function _optics(d::AbstractBatchedNormal)
    return map(c -> VarNames.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end
Plaice.optic_vec(d::AbstractBatchedNormal) = _optics(d)
Plaice.unconstrained_optic_vec(d::AbstractBatchedNormal) = _optics(d)
