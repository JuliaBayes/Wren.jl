"""
    TDist(ν)

Student's t distribution with `ν` degrees of freedom. The density is

```math
p(x) = \\frac{\\Gamma\\!\\left(\\frac{\\nu+1}{2}\\right)}
             {\\Gamma\\!\\left(\\frac{\\nu}{2}\\right)\\sqrt{\\nu\\pi}}
       \\left(1 + \\frac{x^{2}}{\\nu}\\right)^{-\\frac{\\nu+1}{2}}.
```

This is a standard (zero-location, unit-scale) t distribution. For location-scale
variants, use `LocationScale(μ, σ, TDist(ν))` (not yet implemented).

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct TDist{N<:Number} <: UnconstrainedUnivariateDist
    ν::N
end

@inline function _valuetype(d::TDist, x::Number)
    return float(promote_type(typeof(d.ν), typeof(x)))
end

function checkparams(d::TDist)
    return isfinite(d.ν) & (d.ν > zero(d.ν))
end

@inline function DensityInterface.logdensityof(d::TDist, x::Number)
    T = _valuetype(d, x)
    ν = convert(T, d.ν)
    y = convert(T, x)
    half = (ν + one(T)) / 2
    lg_num = select(half > zero(T), () -> loggamma(half), () -> convert(T, NaN))
    lg_den = select(ν / 2 > zero(T), () -> loggamma(ν / 2), () -> convert(T, NaN))
    lognorm = lg_num - lg_den - (log(ν) + convert(T, logπ)) / 2
    return lognorm - half * log(one(T) + y^2 / ν)
end

#=
  Sample via the ratio: if `z ~ Normal(0,1)` and `v ~ Gamma(ν/2, 1)` then
  `z / sqrt(v / (ν/2))` has a t(ν) distribution. The Gamma draw goes through the
  quantile so automatic differentiation cannot follow `ν`, but `ν` is typically not
  differentiated in practice.
=#
@inline function Base.rand(rng::AbstractRNG, d::TDist)
    F = noisetype(d)
    ν = convert(F, d.ν)
    z = randn(rng, F)
    u = rand(rng, F)
    logv = gammalogquantile(ν / 2, u)
    return z / sqrt(exp(logv) / (ν / 2))
end
