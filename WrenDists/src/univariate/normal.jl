struct Normal{M<:Number,S<:Number} <: UnconstrainedUnivariateDist
    μ::M
    σ::S
end

Normal() = Normal(0.0, 1.0)

@inline zval(d::Normal, x::Number) = (x - d.μ) / d.σ
@inline xval(d::Normal, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Normal, x::Number)
    z = zval(d, x)
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return -(z^2 + log2π) / 2 - logt(σ)
end

# Apply `μ` and `σ` after drawing noise so automatic differentiation can follow them.
@inline function Base.rand(rng::AbstractRNG, d::Normal)
    return xval(d, randn(rng, noisetype(d)))
end
