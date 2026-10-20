"""
    NegativeBinomial(r, p)
    NegativeBinomial(r)
    NegativeBinomial()

Negative binomial distribution counting failures before `r` successes, with success
probability `p` (default 0.5). The shape `r` (default 1.0) may be any positive number.

**Customised batched implementations**: Yes
"""
struct NegativeBinomial{R<:Number,P<:Number} <: DiscreteUnivariateDist
    r::R
    p::P
end

NegativeBinomial(r::Number) = NegativeBinomial(r, one(r) / 2)
NegativeBinomial() = NegativeBinomial(1.0, 0.5)

@inline function insupport(::NegativeBinomial, x::Number)
    return isinteger(x) & (x >= zero(x))
end

@inline function DensityInterface.logdensityof(d::NegativeBinomial, x::Number)
    T = masstype(d, x)
    r, p, k = convert(T, d.r), convert(T, d.p), convert(T, x)
    return select(
        insupport(d, x),
        () -> select(
            k == zero(T),
            () -> r * log(p),
            # The beta form avoids subtracting large loggamma values. At zero its
            # normalising terms cancel analytically, so use the exact value above.
            () -> r * log(p) + xlog1py(k, -p) - log(k + r) - _logbeta(r, k + one(T)),
        ),
        () -> convert(T, -Inf),
    )
end

# A Gamma-Poisson mixture also supports noninteger shapes.
@inline function Base.rand(rng::AbstractRNG, d::NegativeBinomial)
    return select(
        d.p == one(d.p),
        () -> zero(eltype(d)),
        () -> rand(rng, Poisson(rand(rng, Gamma(d.r, (one(d.p) - d.p) / d.p)))),
    )
end

# --- Batched NegativeBinomial ---

struct BatchedNegativeBinomial{R<:AbstractArray,P<:AbstractArray} <:
       DiscreteBatchedUnivariate
    r::R
    p::P
end
Base.size(d::BatchedNegativeBinomial) = size(d.r)
Base.rand(rng::AbstractRNG, d::BatchedNegativeBinomial) =
    rand.(Ref(rng), NegativeBinomial.(d.r, d.p))
function DensityInterface.logdensityof(d::BatchedNegativeBinomial, x)
    r, p = d.r, d.p
    T = float(promote_type(eltype(x), eltype(r), eltype(p)))
    return if within_compile()
        k = @. max(x, zero(x))
        sum(
            @. ifelse(
                isinteger(x) & (x >= zero(x)),
                r * log(p) + xlog1py(k, ifelse(k == zero(k), zero(p), -p)) -
                ifelse(k == zero(k), zero(k), log(k + r) + _logbeta(r, k + one(k))),
                convert(T, -Inf),
            )
        )
    else
        s = zero(T)
        for i in eachindex(x, r, p)
            s += logdensityof(NegativeBinomial(r[i], p[i]), x[i])
        end
        s
    end
end
