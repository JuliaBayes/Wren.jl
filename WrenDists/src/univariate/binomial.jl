"""
    Binomial(n, p)

Binomial distribution for `n` trials with success probability `p`.
"""
struct Binomial{N<:Integer,P<:Number} <: DiscreteUnivariateDist
    n::N
    p::P
end

Base.eltype(::Type{Binomial{N,P}}) where {N,P} = float(P)

@inline function insupport(d::Binomial, x::Number)
    return isinteger(x) & (x >= zero(x)) & (x <= d.n)
end

@inline function logbinom(n::T, k::T) where {T<:Number}
    nc, kc, mc = max(n, zero(T)), max(k, zero(T)), max(n - k, zero(T))
    return loggamma(nc + one(T)) - loggamma(kc + one(T)) - loggamma(mc + one(T))
end

@inline function DensityInterface.logdensityof(d::Binomial, x::Number)
    T = masstype(d, x)
    n, k, p = convert(T, d.n), convert(T, x), convert(T, d.p)
    # Skip zero terms so `p = 0` and `p = 1` work.
    a = select(k == zero(T), () -> zero(T), () -> k * log(p))
    b = select(k == n, () -> zero(T), () -> (n - k) * log1p(-p))
    return select(insupport(d, x), () -> logbinom(n, k) + a + b, () -> convert(T, -Inf))
end

# Add `n` Bernoulli samples. This fixed-length loop also works with tracing tools and
# on GPUs.
@inline function Base.rand(rng::AbstractRNG, d::Binomial)
    T = eltype(d)
    s = zero(T)
    for _ in 1:(d.n)
        u = rand(rng, noisetype(d))
        s += select(u < d.p, () -> one(T), () -> zero(T))
    end
    return s
end

# --- Batched Binomial ---

struct BatchedBinomial{N<:AbstractArray,P<:AbstractArray} <: DiscreteBatchedUnivariate
    n::N
    p::P
end
Base.size(d::BatchedBinomial) = size(d.p)
Base.rand(rng::AbstractRNG, d::BatchedBinomial) = rand.(Ref(rng), Binomial.(d.n, d.p))
function DensityInterface.logdensityof(d::BatchedBinomial, x)
    # Benchmarks @ https://github.com/JuliaBayes/Wren.jl/issues/103
    n, p = d.n, d.p
    return if within_compile()
        sum(@. logbinom(n, x) + xlogy(x, p) + xlogy(n - x, one(x) - p))
    else
        T = float(promote_type(eltype(x), eltype(n), eltype(p)))
        s = zero(T)
        for i in eachindex(x)
            ni, ki, pi = convert(T, n[i]), convert(T, x[i]), convert(T, p[i])
            s += logbinom(ni, ki) + xlogy(ki, pi) + xlogy(ni - ki, one(T) - pi)
        end
        s
    end
end
