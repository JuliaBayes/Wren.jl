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

**Customised batched implementations**: Yes
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

# --- Batched TDist ---

using FillArrays: Fill, AbstractFill, getindex_value

struct BatchedTDist{V<:AbstractArray,M<:AbstractArray,S<:AbstractArray} <: BatchedUnivariate
    ν::V
    μ::M
    σ::S
end
Base.size(d::BatchedTDist) = size(d.μ)

function Base.rand(rng::AbstractRNG, d::BatchedTDist)
    F = noisetype(d)
    N = length(d.μ)
    z = randn(rng, F, N)
    u = rand(rng, F, N)
    logv = gammalogquantile.(d.ν ./ 2, u)
    t = z ./ sqrt.(exp.(logv) ./ (d.ν ./ 2))
    return d.μ .+ d.σ .* t
end

# When ν is shared (AbstractFill), compute the loggamma normalising terms once.
function _tlognorm_sum(ν::AbstractFill, N::Int)
    T = float(eltype(ν))
    ν1 = convert(T, getindex_value(ν))
    half = (ν1 + one(T)) / 2
    return N * (loggamma(half) - loggamma(ν1 / 2) - (log(ν1) + T(logπ)) / 2)
end

# When ν varies, sum the per-element normalising terms.
function _tlognorm_sum(ν::AbstractArray, N::Int)
    T = float(eltype(ν))
    s = zero(T)
    for i in eachindex(ν)
        νi = convert(T, ν[i])
        half = (νi + one(T)) / 2
        s += loggamma(half) - loggamma(νi / 2) - (log(νi) + T(logπ)) / 2
    end
    return s
end

function _tdist_half(ν::AbstractFill)
    T = float(eltype(ν))
    return Fill((convert(T, getindex_value(ν)) + one(T)) / 2, size(ν))
end
_tdist_half(ν::AbstractArray) = (float.(ν) .+ one(float(eltype(ν)))) ./ 2

function DensityInterface.logdensityof(d::BatchedTDist, x)
    # Benchmarks @ bench/batch/gpu/tdist.jl
    ν, μ, σ = d.ν, d.μ, d.σ
    lognorm = _tlognorm_sum(ν, length(x))
    return if within_compile()
        half = _tdist_half(ν)
        z = @. (x - μ) / σ
        sum(@. -log(σ) - half * log(one(eltype(x)) + z^2 / ν)) + lognorm
    else
        T = float(promote_type(eltype(x), eltype(ν), eltype(μ), eltype(σ)))
        half = _tdist_half(ν)
        s = zero(T)
        for i in eachindex(x, μ, σ, half)
            z = (convert(T, x[i]) - convert(T, μ[i])) / convert(T, σ[i])
            s -=
                log(convert(T, σ[i])) +
                convert(T, half[i]) * log(one(T) + z^2 / convert(T, ν[i]))
        end
        s + lognorm
    end
end
