"""
    Exponential(θ)
    Exponential()

Exponential distribution with scale `θ` (default 1.0).
"""
struct Exponential{R<:Number} <: PositiveUnivariateDist
    θ::R
end

Exponential() = Exponential(1.0)

@inline sval(d::Exponential, x::Number) = x / d.θ

@inline function DensityInterface.logdensityof(d::Exponential, x::Number)
    s = sval(d, x)
    θ = oftype(s, d.θ)
    return select(x >= zero(x), () -> -(s + log(θ)), () -> oftype(float(s), -Inf))
end

# Apply `θ` after drawing noise so automatic differentiation can follow it.
@inline function Base.rand(rng::AbstractRNG, d::Exponential)
    e = -log(rand(rng, noisetype(d)))
    return d.θ * e
end

# --- Batched Exponential ---

struct BatchedExponential{R<:AbstractArray} <: BatchedUnivariate
    θ::R
end
Base.size(d::BatchedExponential) = size(d.θ)
function Base.rand(rng::AbstractRNG, d::BatchedExponential)
    return -log.(rand(rng, noisetype(d), size(d))) .* d.θ
end
function DensityInterface.logdensityof(d::BatchedExponential, x)
    s = _zero_logdensity(d, x)
    @inbounds for i in eachindex(x, d.θ)
        s -= x[i] / d.θ[i] + log(d.θ[i])
    end
    return s
end
Plaice.scalar_to_scalar_bijector(::BatchedExponential) = Plaice.Log(0.0, 1)
