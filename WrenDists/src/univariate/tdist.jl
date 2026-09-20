"""
    TDist(ν, μ, σ)
    TDist(ν, μ)
    TDist(ν)

Student's t distribution with `ν` degrees of freedom, location `μ` (default 0.0) and
scale `σ` (default 1.0). The density is

```math
p(x) = \\frac{\\Gamma\\!\\left(\\frac{\\nu+1}{2}\\right)}
             {\\Gamma\\!\\left(\\frac{\\nu}{2}\\right)\\sqrt{\\nu\\pi}\\,\\sigma}
       \\left(1 + \\frac{1}{\\nu}\\left(\\frac{x-\\mu}{\\sigma}\\right)^{2}\\right)
       ^{-\\frac{\\nu+1}{2}}.
```

`σ` is a scale parameter, not a standard deviation: the variance is
``σ^2 ν / (ν - 2)`` and exists only for ``ν > 2``.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct TDist{N<:Number,M<:Number,S<:Number} <: UnconstrainedUnivariateDist
    ν::N
    μ::M
    σ::S
end

TDist(ν::Number, μ::Number) = TDist(ν, μ, one(μ))
TDist(ν::Number) = TDist(ν, 0.0, 1.0)

@inline function _valuetype(d::TDist, x::Number)
    return float(promote_type(typeof(d.ν), typeof(d.μ), typeof(d.σ), typeof(x)))
end

function checkparams(d::TDist)
    return isfinite(d.ν) & (d.ν > zero(d.ν)) & isfinite(d.μ) & isfinite(d.σ) &
           (d.σ > zero(d.σ))
end

@inline zval(d::TDist, x::Number) = (x - d.μ) / d.σ
@inline xval(d::TDist, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::TDist, x::Number)
    T = _valuetype(d, x)
    ν = convert(T, d.ν)
    z = convert(T, zval(d, x))
    σ = convert(T, d.σ)
    half = (ν + one(T)) / 2
    lg_num = select(half > zero(T), () -> loggamma(half), () -> convert(T, NaN))
    lg_den = select(ν / 2 > zero(T), () -> loggamma(ν / 2), () -> convert(T, NaN))
    lognorm = lg_num - lg_den - (log(ν) + convert(T, logπ)) / 2
    return lognorm - log(σ) - half * log(one(T) + z^2 / ν)
end

#=
  Sample via the ratio: if `z ~ Normal(0,1)` and `v ~ Gamma(ν/2, 1)` then
  `z / sqrt(v / (ν/2))` has a standard t(ν) distribution. Location and scale are
  applied afterwards so automatic differentiation can follow them.
=#
@inline function Base.rand(rng::AbstractRNG, d::TDist)
    F = noisetype(d)
    ν = convert(F, d.ν)
    z = randn(rng, F)
    u = rand(rng, F)
    logv = gammalogquantile(ν / 2, u)
    t = z / sqrt(exp(logv) / (ν / 2))
    return xval(d, t)
end
