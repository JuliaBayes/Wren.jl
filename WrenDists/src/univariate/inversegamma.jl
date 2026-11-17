"""
    InverseGamma(α, θ)
    InverseGamma(α)
    InverseGamma()

Inverse gamma distribution with shape `α` (default 1.0) and scale `θ` (default 1.0).
If `X ~ Gamma(α, 1/θ)`, then `1/X ~ InverseGamma(α, θ)`.

**Customised batched implementations**: Yes
"""
struct InverseGamma{A<:Number,T<:Number} <: PositiveUnivariateDist
    α::A
    θ::T
end

InverseGamma(α::Number) = InverseGamma(α, one(α))
InverseGamma() = InverseGamma(1.0, 1.0)

@inline function insupport(d::InverseGamma, x::Number)
    return isfinite(x) & (x > zero(x))
end

@inline function DensityInterface.logdensityof(d::InverseGamma, x::Number)
    T = masstype(d, x)
    α, θ, y = convert(T, d.α), convert(T, d.θ), convert(T, x)
    # `loggamma` throws for a non-positive argument, so an invalid shape takes `NaN`.
    lg = select(α > zero(T), () -> loggamma(α), () -> convert(T, NaN))
    # `abs` keeps `log` from throwing on a negative `x`, which takes `-Inf` below.
    value = α * log(θ) - lg - muladd(α + one(T), log(abs(y)), θ / y)
    return select(insupport(d, y), () -> value, () -> convert(T, -Inf))
end

# The reciprocal of a `Gamma(α, 1/θ)` draw inherits its parameter-differentiable quantile.
@inline function Base.rand(rng::AbstractRNG, d::InverseGamma)
    return inv(_gamma_quantile(Gamma(d.α, inv(d.θ)), rand(rng, noisetype(d))))
end

# --- Batched InverseGamma ---

struct BatchedInverseGamma{A<:AbstractArray,T<:AbstractArray} <: BatchedUnivariate
    α::A
    θ::T
end
Base.size(d::BatchedInverseGamma) = size(d.α)
function Base.rand(rng::AbstractRNG, d::BatchedInverseGamma)
    return rand.(Ref(rng), InverseGamma.(d.α, d.θ))
end
function DensityInterface.logdensityof(d::BatchedInverseGamma, x)
    α, θ = d.α, d.θ
    T = float(promote_type(eltype(x), eltype(α), eltype(θ)))
    return if within_compile()
        sum(
            @. ifelse(
                isfinite(x) & (x > zero(x)),
                α * log(θ) - loggamma(α) - (α + one(α)) * log(abs(x)) - θ / x,
                convert(T, -Inf),
            )
        )
    else
        s = zero(T)
        for i in eachindex(x, α, θ)
            ai, ti, xi = convert(T, α[i]), convert(T, θ[i]), convert(T, x[i])
            value = ai * log(ti) - loggamma(ai) - muladd(ai + one(T), log(abs(xi)), ti / xi)
            s += ifelse(isfinite(xi) & (xi > zero(T)), value, convert(T, -Inf))
        end
        s
    end
end
Plaice.scalar_to_scalar_bijector(::BatchedInverseGamma) = Plaice.Log(0.0, 1)
