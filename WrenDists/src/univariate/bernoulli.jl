"""
    Bernoulli(p)
    Bernoulli()

Bernoulli distribution with success probability `p` (default 0.5).
"""
struct Bernoulli{P<:Number} <: DiscreteUnivariateDist
    p::P
end

Bernoulli() = Bernoulli(0.5)

Base.eltype(::Type{Bernoulli{P}}) where {P} = float(P)

@inline function DensityInterface.logdensityof(d::Bernoulli, x::Number)
    T = masstype(d, x)
    p = convert(T, d.p)
    return select(
        x == one(x),
        () -> log(p),
        () -> select(x == zero(x), () -> log1p(-p), () -> convert(T, -Inf)),
    )
end

# A sample changes in steps as `p` changes, so its derivative is zero almost everywhere.
@inline function Base.rand(rng::AbstractRNG, d::Bernoulli)
    T = eltype(d)
    u = rand(rng, noisetype(d))
    return select(u < d.p, () -> one(T), () -> zero(T))
end

# --- Batched Bernoulli ---

struct BatchedBernoulli{P<:AbstractArray} <: DiscreteBatchedUnivariate
    p::P
end
Base.size(d::BatchedBernoulli) = size(d.p)
function Base.rand(rng::AbstractRNG, d::BatchedBernoulli)
    T = float(eltype(d.p))
    return T.(rand(rng, noisetype(d), size(d)) .< d.p)
end
function DensityInterface.logdensityof(d::BatchedBernoulli, x)
    s = _zero_logdensity(d, x)
    T = typeof(s)
    @inbounds for i in eachindex(x, d.p)
        xi = convert(T, x[i])
        p = convert(T, d.p[i])
        s += ifelse(xi == one(T), log(p), ifelse(xi == zero(T), log1p(-p), T(-Inf)))
    end
    return s
end
