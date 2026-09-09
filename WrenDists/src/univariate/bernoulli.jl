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
        () -> logt(p),
        () -> select(x == zero(x), () -> log1pt(-p), () -> convert(T, -Inf)),
    )
end

# A sample changes in steps as `p` changes, so its derivative is zero almost everywhere.
@inline function Base.rand(rng::AbstractRNG, d::Bernoulli)
    T = eltype(d)
    u = rand(rng, noisetype(d))
    return select(u < d.p, () -> one(T), () -> zero(T))
end
