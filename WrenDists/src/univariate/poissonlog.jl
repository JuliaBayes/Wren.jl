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

@inline function DensityInterface.logdensityof(d::PoissonLog, x::Number)
    T = masstype(d, x)
    logλ, k = convert(T, d.logλ), convert(T, x)
    # Define `0 * log(0)` as zero, so `logλ = -Inf` puts all mass on zero.
    a = select(k == zero(T), () -> zero(T), () -> k * logλ)
    # `loggamma` rejects negative integers before the support check runs.
    g = loggamma(max(k, zero(T)) + one(T))
    # If `log(k!)` overflows, the mass is zero; avoid `Inf - Inf`.
    return select(
        insupport(d, x) & isfinite(g),
        () -> a - exp(logλ) - g,
        () -> convert(T, -Inf),
    )
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
        sum(@. x * logλ - exp(logλ) - loggamma(x + one(x)))
    else
        T = float(promote_type(eltype(x), eltype(logλ)))
        s = zero(T)
        for i in eachindex(x, logλ)
            ki, logλi = convert(T, x[i]), convert(T, logλ[i])
            s += ki * logλi - exp(logλi) - loggamma(ki + one(T))
        end
        s
    end
end
