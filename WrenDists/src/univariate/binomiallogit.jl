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

@inline function binomiallogit_logmass(n::Number, k::Number, η::AbstractFloat)
    positive = η >= zero(η)
    a = ifelse(positive, -η, η)
    c = ifelse(positive, n - k, k)
    return logbinom(n, k) + c * ifelse(iszero(c), zero(a), a) - n * log1pexp(a)
end

# Count-only masks keep cached AD tapes valid when the logit changes sign.
@inline function binomiallogit_logmass(n::Number, k::Number, η::Number)
    η = float(η)
    m = n - k
    # Mask before softplus so endpoint counts also handle infinite logits.
    a = ifelse(iszero(k), zero(η), -η)
    b = ifelse(iszero(m), zero(η), η)
    return logbinom(n, k) - k * log1pexp(a) - m * log1pexp(b)
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
        logp = binomiallogit_logmass.(n, k, η)
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
