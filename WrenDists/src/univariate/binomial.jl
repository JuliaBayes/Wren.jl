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
