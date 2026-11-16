"""
    BinomialLogit(n, logitp)
    BinomialLogit(n)
    BinomialLogit()

Binomial distribution for `n` (default 1) trials with success probability
`logistic(logitp)` (default `logitp = 0.0`, i.e. probability 0.5).

**Customised batched implementations**: Yes
"""
struct BinomialLogit{N<:Integer,P<:Number} <: DiscreteUnivariateDist
    n::N
    logitp::P
end

BinomialLogit(n::Integer) = BinomialLogit(n, 0.0)
BinomialLogit() = BinomialLogit(1, 0.0)

Base.eltype(::Type{BinomialLogit{N,P}}) where {N,P} = float(P)

@inline function insupport(d::BinomialLogit, x::Number)
    return isinteger(x) & (x >= zero(x)) & (x <= d.n)
end

# `k log p + (n - k) log(1 - p) = k η - n log1pexp(η)` for `p = logistic(η)`.
@inline function binomiallogit_logmass(n::T, k::T, η::T) where {T<:Number}
    return logbinom(n, k) + k * η - n * log1pexp(η)
end

@inline function DensityInterface.logdensityof(d::BinomialLogit, x::Number)
    T = masstype(d, x)
    n, k, η = convert(T, d.n), convert(T, x), convert(T, d.logitp)
    value = binomiallogit_logmass(n, k, η)
    return select(insupport(d, x), () -> value, () -> convert(T, -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::BinomialLogit)
    return rand(rng, Binomial(d.n, logistic(d.logitp)))
end

# --- Batched BinomialLogit ---

struct BatchedBinomialLogit{N<:AbstractArray,P<:AbstractArray} <: DiscreteBatchedUnivariate
    n::N
    logitp::P
end
Base.size(d::BatchedBinomialLogit) = size(d.logitp)
function Base.rand(rng::AbstractRNG, d::BatchedBinomialLogit)
    return rand.(Ref(rng), BinomialLogit.(d.n, d.logitp))
end
function DensityInterface.logdensityof(d::BatchedBinomialLogit, x)
    n, η = d.n, d.logitp
    return if within_compile()
        valid = @. isinteger(x) & (x >= zero(x)) & (x <= n)
        k = @. ifelse(valid, x, zero(x))
        η = @. ifelse(valid, η, zero(η))
        logp = @. logbinom(n, k) + k * η - n * log1pexp(η)
        sum(@. ifelse(valid, logp, oftype(logp, -Inf)))
    else
        T = float(promote_type(eltype(x), eltype(n), eltype(η)))
        s = zero(T)
        for i in eachindex(x, n, η)
            isinteger(x[i]) && 0 <= x[i] <= n[i] || return convert(T, -Inf)
            ni, ki, ηi = convert(T, n[i]), convert(T, x[i]), convert(T, η[i])
            s += binomiallogit_logmass(ni, ki, ηi)
        end
        s
    end
end
