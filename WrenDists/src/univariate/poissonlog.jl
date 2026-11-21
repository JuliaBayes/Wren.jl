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

@inline function insupport(::PoissonLog, x::Number)
    return isinteger(x) & (x >= zero(x))
end

# The error in Stirling's formula, which is
#
#   1/(12k) - 1/(360k^3) + 1/(1260k^5) + O(k^-7).
#  
# See e.g.
# https://en.wikipedia.org/wiki/Stirling%27s_approximation#Speed_of_convergence_and_error_estimates
# https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf
const STIRLERR_TABLE = ntuple(
    n -> Float64(
        loggamma(big(n) + 1) - (n + big(0.5)) * log(big(n)) + n - log(2 * big(π)) / 2,
    ),
    15,
)
@inline function _stirlerr_series(k::T) where {T}
    ki = inv(k)
    c = (1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188)
    return ki * evalpoly(ki * ki, map(ci -> convert(T, ci), c))
end
@inline function _stirlerr(k::T) where {T<:Base.IEEEFloat}
    # Fast path for Float16/Float32/Float64. When k is >= 16, the O(k^-7) term is tiny, so
    # we can drop it and use the polynomial approximation. For k < 16, we use a precomputed
    # table of values. This is substantially faster than computing the series.
    return k < 16 ? convert(T, STIRLERR_TABLE[Int(k)]) : _stirlerr_series(k)
end
@inline function _stirlerr(k::T) where {T}
    # For other types (e.g. BigFloat), we compute it analytically.
    return loggamma(k + one(T)) - (k + convert(T, 0.5)) * log(k) + k - convert(T, log2π) / 2
end

@inline function _poissonlog_logpmf(logλ, k, stirlerr)
    # note: the k = 0 case was already handled in logdensityof
    logk = log(k)
    δ = logλ - logk
    return -stirlerr - k * (expm1(δ) - δ) - (log2π + logk) / 2
end

# Our PoissonLog density is pretty convoluted for the sake of numerical accuracy. However,
# the requisite derivatives are much simpler! This leads to a quite noticeable speedup. We
# have that
#
# logpmf = k(logλ) - exp(logλ) - log(k!)
#        = k(logλ) - exp(logλ) - log(2πk)/2 - k(log(k)) + k - stirlerr
#
# from which the derivatives below can be obtained.
EnzymeRules.@easy_rule(
    _poissonlog_logpmf(logλ, k, stirlerr),
    (k - exp(logλ), logλ - log(k) - inv(2k), -one(stirlerr)),
)

@inline function DensityInterface.logdensityof(d::PoissonLog, x::Number)
    # The logpmf we're trying to express here is `k * logλ - exp(logλ) - log(k!)`. However,
    # the expression here is liable to catastrophic cancellation when `k` is large and
    # `logλ` is close to `log(k)`. We therefore use an alternative expression. See #319 for
    # the mathematical details.
    T = masstype(d, x)
    logλ, k = convert(T, d.logλ), convert(T, x)
    return if within_compile()
        kp = max(k, one(T))
        lp = ifelse(k == zero(T), -exp(logλ), _poissonlog_logpmf(logλ, kp, _stirlerr(kp)))
        ifelse(insupport(d, x), lp, convert(T, -Inf))
    else
        insupport(d, x) || return convert(T, -Inf)
        # avoid calculating log(k) in _poissonlog_logpmf when k == 0.
        k == zero(T) && return -exp(logλ)
        # If `T` is an AD type e.g. ForwardDiff.Dual, we need to unwrap it to calculate
        # `_stirlerr` on its 'underlying' float type.
        F = typeof(eps(T))
        _poissonlog_logpmf(logλ, k, convert(T, _stirlerr(convert(F, x))))
    end
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
