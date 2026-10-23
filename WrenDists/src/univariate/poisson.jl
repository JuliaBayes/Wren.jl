"""
    Poisson(λ)
    Poisson()

Poisson distribution with rate `λ` (default 1.0).

**Customised batched implementations**: Yes
"""
struct Poisson{L<:Number} <: DiscreteUnivariateDist
    λ::L
end

Poisson() = Poisson(1.0)

Base.eltype(::Type{Poisson{L}}) where {L} = float(L)

checkparams(d::Poisson) = isfinite(d.λ) & (d.λ >= zero(d.λ))

@inline function insupport(d::Poisson, x::Number)
    return isinteger(x) & (x >= zero(x))
end

@inline function horizon(d::Poisson)
    λ = float(d.λ)
    n = λ + 20 * sqrt(max(λ, zero(λ))) + 60
    return ((n > zero(n)) & (n <= typemax(Int))) ? ceil(Int, n) : 0
end

@inline usable(d::Poisson) = checkparams(d) & (horizon(d) > 0)

@inline function DensityInterface.logdensityof(d::Poisson, x::Number)
    T = masstype(d, x)
    λ, k = convert(T, d.λ), convert(T, x)
    # Define `0 * log(0)` as zero.
    a = select(k == zero(T), () -> zero(T), () -> k * log(λ))
    # `loggamma` rejects negative integers before the support check runs.
    g = loggamma(max(k, zero(T)) + one(T))
    # If `log(k!)` overflows, the mass is zero; avoid `Inf - Inf`.
    return select(insupport(d, x) & isfinite(g), () -> a - λ - g, () -> convert(T, -Inf))
end

# --- Poisson quantile machinery (needed by rand) ---

# `P(X ≤ k) = Q(k+1, λ)` and `P(X > k) = P(k+1, λ)`.
@inline function _poisson_tails(::Type{T}, d::Poisson, x::Number) where {T<:AbstractFloat}
    nan = convert(T, NaN)
    usable(d) || return (nan, nan)
    k = floor(convert(T, x))
    isnan(k) && return (nan, nan)
    k < zero(T) && return (zero(T), one(T))
    lower, upper = gamma_inc(k + one(T), convert(T, d.λ))
    return (upper, lower)
end

@inline function _poisson_cdf(::Type{T}, d::Poisson, x::Number) where {T<:AbstractFloat}
    return _poisson_tails(T, d, x)[1]
end

function _poisson_quantile(::Type{T}, d::Poisson, q::Number) where {T<:AbstractFloat}
    usable(d) || return convert(T, NaN)
    p = convert(T, q)
    hi = convert(T, horizon(d))
    p >= one(T) && return hi
    p > zero(T) || return zero(T)
    λ = convert(T, d.λ)
    # A normal approximation with a continuity correction starts within a step or two.
    z = convert(T, -sqrt2 * erfcinvt(2 * p))
    k = clamp(floor(λ + z * sqrt(λ) - one(T) / 2), zero(T), hi)
    while (k > zero(T)) && (_poisson_cdf(T, d, _prevcount(k)) >= p)
        k = _prevcount(k)
    end
    while (k < hi) && (_poisson_cdf(T, d, k) < p)
        k = _nextcount(k)
    end
    return k
end

# Above `maxintfloat(T)`, `k ± 1` rounds back to `k`, so step to the adjacent float instead.
# Every float that large is an integer.
@inline _nextcount(k::AbstractFloat) = max(k + one(k), nextfloat(k))
@inline _prevcount(k::AbstractFloat) = min(k - one(k), prevfloat(k))

# CDF inversion avoids the `exp(-λ)` underflow in product-of-uniforms sampling.
@inline function Base.rand(rng::AbstractRNG, d::Poisson)
    T = noisetype(d)
    return _poisson_quantile(T, d, rand(rng, T))
end

# --- Batched Poisson ---

struct BatchedPoisson{L<:AbstractArray} <: DiscreteBatchedUnivariate
    λ::L
end
Base.size(d::BatchedPoisson) = size(d.λ)
Base.rand(rng::AbstractRNG, d::BatchedPoisson) = rand.(Ref(rng), Poisson.(d.λ))
function DensityInterface.logdensityof(d::BatchedPoisson, x)
    # Benchmarks @ https://github.com/JuliaBayes/Wren.jl/issues/104
    λ = d.λ
    return if within_compile()
        sum(((λi, xi) -> logdensityof(Poisson(λi), xi)).(λ, x))
    else
        T = float(promote_type(eltype(x), eltype(λ)))
        s = zero(T)
        for i in eachindex(x, λ)
            s += logdensityof(Poisson(λ[i]), x[i])
        end
        s
    end
end
