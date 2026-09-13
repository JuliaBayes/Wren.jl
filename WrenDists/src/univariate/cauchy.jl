"""
    Cauchy(μ, σ)
    Cauchy()

Cauchy distribution with location `μ` (default 0.0) and scale `σ` (default 1.0).
"""
struct Cauchy{M<:Number,S<:Number} <: UnconstrainedUnivariateDist
    μ::M
    σ::S
end

Cauchy() = Cauchy(0.0, 1.0)

@inline zval(d::Cauchy, x::Number) = (x - d.μ) / d.σ
@inline xval(d::Cauchy, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Cauchy, x::Number)
    z = zval(d, x)
    σ = oftype(z, d.σ)
    return -log(σ) - logπ - 2 * log(hypot(one(z), z))
end

@inline function Base.rand(rng::AbstractRNG, d::Cauchy)
    u = rand(rng, noisetype(d))
    return xval(d, -cospi(u) / sinpi(u))
end
