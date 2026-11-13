"""
    Bernoulli(p)
    Bernoulli()

Bernoulli distribution with success probability `p` (default 0.5).

**Customised batched implementations**: Yes
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
    # Benchmarks @ https://github.com/JuliaBayes/Wren.jl/issues/102
    p = d.p
    return if within_compile()
        p = convert.(float(promote_type(eltype(p), eltype(x))), p)
        # Compiled selects evaluate both branches; keep inactive logarithms finite.
        sum(
            @. ifelse(
                (x == zero(x)) | (x == one(x)),
                log(ifelse(x == one(x), p, one(p))) +
                log1p(-ifelse(x == zero(x), p, zero(p))),
                oftype(p, -Inf),
            )
        )
    else
        sum(((p, x) -> logdensityof(Bernoulli(p), x)).(p, x))
    end
end
