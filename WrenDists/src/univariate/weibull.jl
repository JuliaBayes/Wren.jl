"""
    Weibull(α, θ)
    Weibull(α)
    Weibull()

Weibull distribution with shape `α` (default 1.0) and scale `θ` (default 1.0).

**Customised batched implementations**: Yes
"""
struct Weibull{A<:Number,T<:Number} <: PositiveUnivariateDist
    α::A
    θ::T
end

Weibull(α) = Weibull(α, one(α))
Weibull() = Weibull(1.0, 1.0)

@inline sval(d::Weibull, x::Number) = x / d.θ

@inline function DensityInterface.logdensityof(d::Weibull, x::Number)
    scaled, α = map(float, promote(sval(d, x), d.α))
    s = abs(scaled)
    # Match `θ` to `s` so integer parameters do not reduce `BigFloat` precision.
    θ = oftype(s, d.θ)
    # At an infinite `s` the log and power terms cancel to `NaN`; the density is zero.
    value = select(
        isinf(s),
        () -> oftype(s, -Inf),
        () -> log(α) - log(θ) + (α - one(α)) * log(s) - s^α,
    )
    # At `x = 0` the formula settles every shape but `α = 1`, where `0 * log(0)` is `NaN`.
    atzero = select(α == one(α), () -> -log(θ), () -> value)
    return select(
        x > zero(x),
        () -> value,
        () -> select(x == zero(x), () -> atzero, () -> oftype(s, -Inf)),
    )
end

#=
  A draw is `θ E^(1/α)` for a unit exponential `E`. Applying the parameters after
  drawing noise lets automatic differentiation follow them.
=#
@inline function Base.rand(rng::AbstractRNG, d::Weibull)
    e = -log(rand(rng, noisetype(d)))
    α, θ = promote(d.α, d.θ)
    return θ * e^inv(α)
end

# --- Batched Weibull ---

struct BatchedWeibull{A<:AbstractArray,T<:AbstractArray} <: BatchedUnivariate
    α::A
    θ::T
end
Base.size(d::BatchedWeibull) = size(d.α)
Base.rand(rng::AbstractRNG, d::BatchedWeibull) = rand.(Ref(rng), Weibull.(d.α, d.θ))
function DensityInterface.logdensityof(d::BatchedWeibull, x)
    α, θ = d.α, d.θ
    T = float(promote_type(eltype(x), eltype(α), eltype(θ)))
    return if within_compile()
        z = @. abs(x / θ)
        sum(
            @. ifelse(
                (x >= zero(x)) & !isinf(z),
                ifelse(
                    (x == zero(x)) & (α == one(α)),
                    -log(θ),
                    log(α) - log(θ) + (α - one(α)) * log(z) - z^α,
                ),
                convert(T, -Inf),
            )
        )
    else
        s = zero(T)
        for i in eachindex(x, α, θ)
            ai, ti, xi = convert(T, α[i]), convert(T, θ[i]), convert(T, x[i])
            z, lt = abs(xi / ti), log(ti)
            value = log(ai) - lt + (ai - one(T)) * log(z) - z^ai
            value = ifelse((xi == zero(T)) & (ai == one(T)), -lt, value)
            s += ifelse((xi >= zero(T)) & !isinf(z), value, convert(T, -Inf))
        end
        s
    end
end
Plaice.scalar_to_scalar_bijector(d::BatchedWeibull) = Plaice.Log(zero(noisetype(d)), 1)
