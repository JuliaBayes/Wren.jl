"""
    BernoulliLogit(logitp)
    BernoulliLogit()

Bernoulli distribution with success probability `logistic(logitp)` (default `logitp = 0.0`,
i.e. probability 0.5).

**Customised batched implementations**: Yes
"""
struct BernoulliLogit{P<:Number} <: DiscreteUnivariateDist
    logitp::P
end

BernoulliLogit() = BernoulliLogit(0.0)

Base.eltype(::Type{BernoulliLogit{P}}) where {P} = float(P)

# `log(logistic(η)) = -log1pexp(-η)` and `log(1 - logistic(η)) = -log1pexp(η)`.
@inline function DensityInterface.logdensityof(d::BernoulliLogit, x::Number)
    T = masstype(d, x)
    η = convert(T, d.logitp)
    return select(
        x == one(x),
        () -> -log1pexp(-η),
        () -> select(x == zero(x), () -> -log1pexp(η), () -> convert(T, -Inf)),
    )
end

@inline Base.rand(rng::AbstractRNG, d::BernoulliLogit) =
    rand(rng, Bernoulli(logistic(d.logitp)))

# --- Batched BernoulliLogit ---

struct BatchedBernoulliLogit{P<:AbstractArray} <: DiscreteBatchedUnivariate
    logitp::P
end
Base.size(d::BatchedBernoulliLogit) = size(d.logitp)
function Base.rand(rng::AbstractRNG, d::BatchedBernoulliLogit)
    return rand(rng, BatchedBernoulli(logistic.(d.logitp)))
end
function DensityInterface.logdensityof(d::BatchedBernoulliLogit, x)
    η = d.logitp
    return if within_compile()
        sum(@. x * η - log1pexp(η))
    else
        T = float(promote_type(eltype(x), eltype(η)))
        s = zero(T)
        for i in eachindex(x, η)
            ηi = convert(T, η[i])
            s -= x[i] == one(x[i]) ? log1pexp(-ηi) : log1pexp(ηi)
        end
        s
    end
end
