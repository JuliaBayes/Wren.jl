"""
    Logistic(μ, θ)
    Logistic(μ)
    Logistic()

Logistic distribution with location `μ` (default 0.0) and scale `θ` (default 1.0).

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct Logistic{M<:Number,T<:Number} <: UnconstrainedUnivariateDist
    μ::M
    θ::T
end

Logistic(μ) = Logistic(μ, one(μ))
Logistic() = Logistic(0.0, 1.0)

@inline zval(d::Logistic, x::Number) = (x - d.μ) / d.θ
@inline xval(d::Logistic, z::Number) = muladd(d.θ, z, d.μ)

# The density is symmetric in `z`. Using `-|z|` keeps `log1pexp` in its accurate range and
# makes both tails finite.
@inline function DensityInterface.logdensityof(d::Logistic, x::Number)
    z = zval(d, x)
    θ = oftype(z, d.θ)
    u = -abs(z)
    return u - 2 * log1pexp(u) - log(θ)
end

# Inverse CDF: the logit of a uniform draw follows Logistic(0, 1).
@inline function Base.rand(rng::AbstractRNG, d::Logistic)
    u = rand(rng, noisetype(d))
    return xval(d, log(u) - log1p(-u))
end
