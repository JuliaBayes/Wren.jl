"""
    Geometric(p)
    Geometric()

Geometric distribution with success probability `p` (default 0.5), counting the number of
failures before the first success.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct Geometric{P<:Number} <: DiscreteUnivariateDist
    p::P
end

Geometric() = Geometric(0.5)

Base.eltype(::Type{Geometric{P}}) where {P} = float(P)

@inline function insupport(d::Geometric, x::Number)
    return isinteger(x) & (x >= zero(x))
end

@inline function DensityInterface.logdensityof(d::Geometric, x::Number)
    T = masstype(d, x)
    k, p = convert(T, x), convert(T, d.p)
    # At `p == 1` the tail factor is `0 * -Inf`. Define it as zero.
    tail = select(k == zero(T), () -> zero(T), () -> k * log1p(-p))
    return select(insupport(d, x), () -> log(p) + tail, () -> convert(T, -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::Geometric)
    T = eltype(d)
    u = rand(rng, noisetype(d))
    p = convert(T, d.p)
    # `log1p(-u)` rather than `log(1 - u)`, so an exact zero draw gives zero, not `Inf`.
    return floor(log1p(-u) / log1p(-p))
end
