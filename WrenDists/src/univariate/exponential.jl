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
