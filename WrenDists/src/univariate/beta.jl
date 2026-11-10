"""
    Beta(α, β)
    Beta()

Beta distribution with shape parameters `α` (default 1.0) and `β` (default 1.0).

**Customised batched implementations**: Yes
"""
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
    value = xlogy(α - one(T), y) + xlogy(β - one(T), one(T) - y) - logbeta(α, β)
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

# --- Batched Beta ---

struct BatchedBeta{A<:AbstractArray,B<:AbstractArray} <: BatchedUnivariate
    α::A
    β::B
end
Base.size(d::BatchedBeta) = size(d.α)
Base.rand(rng::AbstractRNG, d::BatchedBeta) = rand.(Ref(rng), Beta.(d.α, d.β))

function _beta_loop(α, β, x)
    T = float(promote_type(eltype(x), eltype(α), eltype(β)))
    s = zero(T)
    for i in eachindex(x, α, β)
        ai, bi, xi = convert(T, α[i]), convert(T, β[i]), convert(T, x[i])
        lx = log(ifelse((xi == zero(T)) & (ai == one(T)), one(T), xi))
        ly = log1p(-ifelse((xi == one(T)) & (bi == one(T)), zero(T), xi))
        value = (ai - one(T)) * lx + (bi - one(T)) * ly - _logbeta(ai, bi)
        d = Beta(ai, bi)
        s += ifelse(insupport(d, xi) & checkparams(d), value, convert(T, -Inf))
    end
    return s
end

# The rule returns dense tangents. Differentiate the loop for Fill and other wrappers.
EnzymeRules.@easy_rule(
    _beta_loop(α::Array, β::Array, x::Array),
    @setup(ψ=digamma.(α .+ β)),
    (
        log.(x) .- digamma.(α) .+ ψ,
        log1p.(-x) .- digamma.(β) .+ ψ,
        (α .- 1) ./ ifelse.(α .== 1, one.(x), x) .-
        (β .- 1) ./ ifelse.(β .== 1, one.(x), 1 .- x),
    ),
)

function DensityInterface.logdensityof(d::BatchedBeta, x)
    α, β = d.α, d.β
    return if within_compile()
        T = float(promote_type(eltype(x), eltype(α), eltype(β)))
        # Avoid xlogy/xlog1py's zero-coefficient branch, which loses shape gradients.
        lx = @. log(ifelse((x == zero(x)) & (α == one(α)), one(x), x))
        ly = @. log1p(-ifelse((x == one(x)) & (β == one(β)), zero(x), x))
        sum(
            @. ifelse(
                isfinite(x) & (x >= zero(x)) & (one(x) - x >= zero(x)) & isfinite(α) &
                isfinite(β) & (α > zero(α)) & (β > zero(β)),
                (α - one(α)) * lx + (β - one(β)) * ly - _logbeta(α, β),
                convert(T, -Inf),
            )
        )
    else
        _beta_loop(α, β, x)
    end
end
Plaice.scalar_to_scalar_bijector(d::BatchedBeta) =
    Plaice.Untruncate(zero(noisetype(d)), one(noisetype(d)))
