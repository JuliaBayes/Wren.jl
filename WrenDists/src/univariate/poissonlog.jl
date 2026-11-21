"""
    PoissonLog(logλ)
    PoissonLog()

Poisson distribution with rate `exp(logλ)` (default `logλ = 0.0`, i.e. rate 1.0).

**Customised batched implementations**: Yes
"""
struct PoissonLog{L<:Number} <: DiscreteUnivariateDist
    logλ::L
end

PoissonLog() = PoissonLog(0.0)

Base.eltype(::Type{PoissonLog{L}}) where {L} = float(L)

@inline function insupport(d::PoissonLog, x::Number)
    return isinteger(x) & (x >= zero(x))
end

# `log(k!) - [(k + 1/2) log(k) - k + log(2π)/2]`, i.e. the error in Stirling's formula.
@inline function _stirlerr(k::T) where {T}
    return select(k < convert(T, 16), () -> _stirlerr_direct(k), () -> _stirlerr_series(k))
end

# Below 16 the direct difference loses little precision. `min` keeps `loggamma` finite
# when both branches are evaluated (e.g. under Reactant).
@inline function _stirlerr_direct(k::T) where {T}
    ks = min(k, convert(T, 16))
    return loggamma(ks + one(T)) - (ks + convert(T, 0.5)) * log(ks) + ks -
           convert(T, log2π) / 2
end

# Asymptotic series; for k >= 16 the first omitted term is below 1e-20.
@inline function _stirlerr_series(k::T) where {T}
    ki = inv(k)
    c = (1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188)
    return ki * evalpoly(ki * ki, map(ci -> convert(T, ci), c))
end

@inline function _poissonlog_pos(logλ, k)
    logk = log(k)
    δ = logλ - logk
    return -_stirlerr(k) - k * (expm1(δ) - δ) - (log2π + logk) / 2
end

# The derivatives are those of `k * logλ - exp(logλ) - loggamma(k + 1)`. Enzyme is slower
# when it differentiates through the expression above.
EnzymeRules.@easy_rule(_poissonlog_pos(logλ, k), (k - exp(logλ), logλ - digamma(k + 1)))

@inline function DensityInterface.logdensityof(d::PoissonLog, x::Number)
    T = masstype(d, x)
    logλ, k = convert(T, d.logλ), convert(T, x)
    # Computing `k * logλ - λ - log(k!)` directly cancels badly when k and λ are large,
    # since all three terms are huge but their sum is not. Instead use Loader (2000):
    #   log p(k) = -stirlerr(k) - bd0(k, λ) - log(2πk)/2,
    # where bd0(k, λ) = k log(k/λ) + λ - k = k * (exp(δ) - 1 - δ) with δ = logλ - log(k).
    # This also never forms `log(k!)`, so nothing overflows for huge k.
    # `max` keeps this branch finite when k = 0 (and for k < 0, before the support check).
    # It is computed inside the closure so reverse-mode AD does not record it when k = 0.
    kp = max(k, one(T))
    # k = 0 gives `-λ`, which also handles `logλ = -Inf` putting all mass on zero.
    lp = select(k == zero(T), () -> -exp(logλ), () -> _poissonlog_pos(logλ, kp))
    return select(insupport(d, x), () -> lp, () -> convert(T, -Inf))
end

@inline Base.rand(rng::AbstractRNG, d::PoissonLog) = rand(rng, Poisson(exp(d.logλ)))

# --- Batched PoissonLog ---

struct BatchedPoissonLog{L<:AbstractArray} <: DiscreteBatchedUnivariate
    logλ::L
end
Base.size(d::BatchedPoissonLog) = size(d.logλ)
Base.rand(rng::AbstractRNG, d::BatchedPoissonLog) = rand.(Ref(rng), PoissonLog.(d.logλ))
function DensityInterface.logdensityof(d::BatchedPoissonLog, x)
    logλ = d.logλ
    return if within_compile()
        sum(((logλi, xi) -> logdensityof(PoissonLog(logλi), xi)).(logλ, x))
    else
        T = float(promote_type(eltype(x), eltype(logλ)))
        s = zero(T)
        for i in eachindex(x, logλ)
            s += logdensityof(PoissonLog(logλ[i]), x[i])
        end
        s
    end
end
