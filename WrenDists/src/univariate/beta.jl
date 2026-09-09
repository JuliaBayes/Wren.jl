struct Beta{A<:Number,B<:Number} <: UnivariateDist
    α::A
    β::B
end

Beta() = Beta(1.0, 1.0)

function checkparams(d::Beta)
    return isfinite(d.α) & isfinite(d.β) & (d.α > zero(d.α)) & (d.β > zero(d.β))
end

@inline function insupport(d::Beta, x::Number)
    return isfinite(x) & (x >= zero(x)) & (one(x) - x >= zero(x))
end

@inline function _valuetype(d::Beta, x::Number)
    return float(promote_type(typeof(d.α), typeof(d.β), typeof(x)))
end

@inline function DensityInterface.logdensityof(d::Beta, x::Number)
    T = _valuetype(d, x)
    α, β, y = convert(T, d.α), convert(T, d.β), convert(T, x)
    value = xlogyt(α - one(T), y) + xlogyt(β - one(T), one(T) - y) - logbetat(α, β)
    return select(insupport(d, y) & checkparams(d), () -> value, () -> convert(T, -Inf))
end

#=
  `X / (X + Y)` for gamma draws `X` and `Y` is `1 / (1 + Y/X)`, and the log quantiles
  give `Y/X` without either draw underflowing. The shapes enter through the quantiles,
  so automatic differentiation follows both.
=#
@inline function Base.rand(rng::AbstractRNG, d::Beta)
    F = noisetype(d)
    α, β = map(float, promote(d.α, d.β))
    valid = checkparams(d)
    # `loggamma` throws for a non-positive shape, so invalid parameters take `NaN`.
    lx = select(valid, () -> gammalogquantile(α, rand(rng, F)), () -> oftype(α, NaN))
    ly = select(valid, () -> gammalogquantile(β, rand(rng, F)), () -> oftype(β, NaN))
    return inv(one(lx) + exp(ly - lx))
end
