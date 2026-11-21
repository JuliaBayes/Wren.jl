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
# For the usual float types, k < 16 is looked up in a table and larger k use the
# asymptotic series. Both are accurate to that type's precision.
@inline function _stirlerr(k::T) where {T<:Base.IEEEFloat}
    return k < 16 ? _stirlerr_table(k) : _stirlerr_series(k)
end

# Other types (e.g. BigFloat) may need more precision than the table and the series give,
# so work it out directly. Any cancellation here is in that type's own precision.
@inline function _stirlerr(k::T) where {T}
    return loggamma(k + one(T)) - (k + convert(T, 0.5)) * log(k) + k - convert(T, log2π) / 2
end

# `stirlerr(n)` for n = 1, ..., 15, worked out in BigFloat and rounded to Float64.
const STIRLERR_TABLE = ntuple(
    n -> Float64(
        loggamma(big(n) + 1) - (n + big(0.5)) * log(big(n)) + n - log(2 * big(π)) / 2,
    ),
    15,
)

# Only whole numbers 1, ..., 15 reach here from `logdensityof`.
@inline _stirlerr_table(k::T) where {T} = convert(T, STIRLERR_TABLE[Int(k)])

# Asymptotic series; for k >= 16 the first omitted term is below 1e-20.
@inline function _stirlerr_series(k::T) where {T}
    ki = inv(k)
    c = (1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188)
    return ki * evalpoly(ki * ki, map(ci -> convert(T, ci), c))
end

# Log-mass for k >= 1, given `s = stirlerr(k)`.
@inline function _poissonlog_pos(logλ, k, s)
    logk = log(k)
    δ = logλ - logk
    return -s - k * (expm1(δ) - δ) - (log2π + logk) / 2
end

# Enzyme is slower when it differentiates through the expression above. The partials are:
#   logλ: k - exp(logλ)
#   k (with s held fixed): δ - 1/(2k), since d/dk [k (exp(δ) - 1 - δ)] = -δ
#   s: -1
EnzymeRules.@easy_rule(
    _poissonlog_pos(logλ, k, s),
    (k - exp(logλ), logλ - log(k) - inv(2k), -one(s)),
)

# Computing `k * logλ - λ - log(k!)` directly cancels badly when k and λ are large, since
# all three terms are huge but their sum is not. Instead use Loader (2000):
#   log p(k) = -stirlerr(k) - bd0(k, λ) - log(2πk)/2,
# where bd0(k, λ) = k log(k/λ) + λ - k = k * (exp(δ) - 1 - δ) with δ = logλ - log(k).
# This also never forms `log(k!)`, so nothing overflows for huge k.
@inline function DensityInterface.logdensityof(d::PoissonLog, x::Number)
    T = masstype(d, x)
    logλ, k = convert(T, d.logλ), convert(T, x)
    within_compile() && return _poissonlog_traced(d, x, logλ, k)
    insupport(d, x) || return convert(T, -Inf)
    # k = 0 gives `-λ`, which also handles `logλ = -Inf` putting all mass on zero.
    k == zero(T) && return -exp(logλ)
    # `stirlerr` only depends on the data, so work it out on the plain float type
    # (`typeof(eps(T))`, as in `rngtype`), where the fast table and series apply even when
    # `T` is an AD number type.
    F = typeof(eps(T))
    return _poissonlog_pos(logλ, k, convert(T, _stirlerr(convert(F, x))))
end

# Under Reactant both sides of each `select` are evaluated, and `x` may be traced.
@inline function _poissonlog_traced(d::PoissonLog, x, logλ::T, k::T) where {T}
    # `max` keeps the k >= 1 branch finite when k = 0, and for k < 0 before the support
    # check (where `loggamma` would also reject negative integers).
    kp = max(k, one(T))
    lp = select(
        k == zero(T),
        () -> -exp(logλ),
        () -> _poissonlog_pos(logλ, kp, _stirlerr(kp)),
    )
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
