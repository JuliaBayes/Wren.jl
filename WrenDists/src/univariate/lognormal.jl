"""
    LogNormal(μ, σ)
    LogNormal()

Log-normal distribution with log-scale mean `μ` (default 0.0) and log-scale standard
deviation `σ` (default 1.0).
"""
struct LogNormal{M<:Number,S<:Number} <: PositiveUnivariateDist
    μ::M
    σ::S
end

LogNormal() = LogNormal(0.0, 1.0)

@inline zval(d::LogNormal, x::Number) = (log(x) - d.μ) / d.σ
@inline xval(d::LogNormal, z::Number) = exp(muladd(d.σ, z, d.μ))

@inline function DensityInterface.logdensityof(d::LogNormal, x::Number)
    lx = log(x)
    z = (lx - d.μ) / d.σ
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return select(
        x > zero(x),
        () -> -(z^2 + log2π) / 2 - log(σ) - lx,
        () -> oftype(float(z), -Inf),
    )
end

# Apply `μ` and `σ` after drawing noise so automatic differentiation can follow them.
@inline Base.rand(rng::AbstractRNG, d::LogNormal) = xval(d, randn(rng, noisetype(d)))
