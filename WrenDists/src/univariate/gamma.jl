"""
    Gamma(α, θ)
    Gamma(α)
    Gamma()

Gamma distribution with shape `α` (default 1.0) and scale `θ` (default 1.0).

**Customised batched implementations**: Yes
"""
struct Gamma{A<:Number,T<:Number} <: PositiveUnivariateDist
    α::A
    θ::T
end

Gamma(α::Number) = Gamma(α, one(α))
Gamma() = Gamma(1.0, 1.0)

@inline function _valuetype(d::Gamma, x::Number)
    return float(promote_type(typeof(d.α), typeof(d.θ), typeof(x)))
end

function checkparams(d::Gamma)
    return isfinite(d.α) & isfinite(d.θ) & (d.α > zero(d.α)) & (d.θ > zero(d.θ))
end

@inline function insupport(d::Gamma, x::Number)
    return isfinite(x) & (x >= zero(x))
end

@inline function DensityInterface.logdensityof(d::Gamma, x::Number)
    T = _valuetype(d, x)
    α, θ, y = convert(T, d.α), convert(T, d.θ), convert(T, x)
    # `loggamma` throws for a non-positive argument, so an invalid shape takes `NaN`.
    lg = select(α > zero(T), () -> loggamma(α), () -> convert(T, NaN))
    value = muladd(α - one(T), log(abs(y)), -(y / θ)) - lg - α * log(θ)
    # At `x = 0` the formula settles every shape but `α = 1`, where `0 * log(0)` is `NaN`.
    atzero = select(α == one(T), () -> -log(θ), () -> value)
    return select(
        insupport(d, y) & (y > zero(T)),
        () -> value,
        () -> select(y == zero(T), () -> atzero, () -> convert(T, -Inf)),
    )
end

# --- Gamma quantile machinery (needed by rand) ---

const GAMMAQUANTILE_MAXITER = 100
const GAMMAQUANTILE_STEPS = 8

gammaquantile(a::Number, p::Number) = exp(gammalogquantile(a, p))

function gammalogquantile(a::Number, p::Number)
    fixedlength(a, p) && return gammalogquantile_fixed(a, p)
    lower = p <= one(p) / 2
    target = lower ? log(p) : log1p(-p)
    lg = loggamma(a)
    u = gammaquantile_start(a, p)
    tol = sqrt(eps(basefloat(typeof(p))))
    for _ in 1:GAMMAQUANTILE_MAXITER
        step = gammaquantile_step(a, lg, u, exp(u), lower, target)
        u -= step
        abs(step) <= tol && break
    end
    #=
      The value is now accurate to the rounding error, but the derivative automatic
      differentiation carries through the iteration lags one step behind. One more step
      brings it to the same accuracy.
    =#
    return u - gammaquantile_step(a, lg, u, exp(u), lower, target)
end

gammaquantile_fixed(a::Number, p::Number) = exp(gammalogquantile_fixed(a, p))

function gammalogquantile_fixed(a::Number, p::Number)
    lower = p <= one(p) / 2
    target = select(lower, () -> log(p), () -> log1p(-p))
    lg = loggamma(a)
    u = gammaquantile_start(a, p)
    for _ in 1:GAMMAQUANTILE_STEPS
        u -= gammaquantile_step(a, lg, u, exp(u), lower, target)
    end
    return u
end

#=
  Far from the root the step can overshoot deep into the upper tail, from where Newton
  crawls back one unit of `u` per step, so the step is bounded. A subnormal `y` carries
  too few digits for the tails to settle, and a zero or infinite one gives `NaN`; the
  quantile is then beyond the normal range and the current `u` is as good as it gets.
=#
@inline function gammaquantile_step(a, lg, u, y, lower, target)
    g = muladd(a - one(a), u, -y) - lg
    tail = select(lower, () -> loggammap(a, y), () -> loggammaq(a, y))
    step = (tail - target) * exp(tail - g - u)
    bounded = min(max(step, -2 * one(step)), 2 * one(step))
    normal = (y > oftype(y, floatmin(basefloat(typeof(y))))) & isfinite(y)
    safe = select(normal, () -> bounded, () -> zero(bounded))
    return select(lower, () -> safe, () -> -safe)
end

#=
  Near zero, `P(a, y) ≈ y^a / Γ(a+1)` inverts directly. Elsewhere, Wilson and
  Hilferty's cube-root normal approximation is good to a few digits.
=#
@inline function gammaquantile_start(a::Number, p::Number)
    logr = (log(p) + loggamma(a + one(a))) / a
    z = -(sqrt2 * erfcinvt(2 * p))
    w = a * (one(a) - inv(9 * a) + z / (3 * sqrt(a)))^3
    small = logr < log((one(a) + a) / (5 * a))
    usable = isfinite(w) & (w > zero(w))
    return select(small | !usable, () -> logr, () -> log(w))
end

function _gamma_quantile(d::Gamma, p::Number)
    T = _valuetype(d, p)
    α, θ, q = convert(T, d.α), convert(T, d.θ), convert(T, p)
    valid = checkparams(d) & (q >= zero(T)) & (q <= one(T))
    inside = (q > zero(T)) & (q < one(T))
    edge = select(q == one(T), () -> convert(T, Inf), () -> zero(T))
    value = select(valid & inside, () -> θ * gammaquantile(α, q), () -> edge)
    return select(valid, () -> value, () -> convert(T, NaN))
end

# The noise is plain, and the parameters enter through `quantile`, so automatic
# differentiation follows them through the Newton solve.
@inline function Base.rand(rng::AbstractRNG, d::Gamma)
    return _gamma_quantile(d, rand(rng, noisetype(d)))
end

# --- Batched Gamma ---

struct BatchedGamma{A<:AbstractArray,T<:AbstractArray} <: BatchedUnivariate
    α::A
    θ::T
end
Base.size(d::BatchedGamma) = size(d.α)
Base.rand(rng::AbstractRNG, d::BatchedGamma) = rand.(Ref(rng), Gamma.(d.α, d.θ))
function DensityInterface.logdensityof(d::BatchedGamma, x)
    α, θ = d.α, d.θ
    T = float(promote_type(eltype(x), eltype(α), eltype(θ)))
    return if within_compile()
        sum(
            @. ifelse(
                isfinite(x) & (x >= zero(x)),
                ifelse(
                    (x == zero(x)) & (α == one(α)),
                    -log(θ),
                    (α - one(α)) * log(abs(x)) - x / θ - loggamma(α) - α * log(θ),
                ),
                convert(T, -Inf),
            )
        )
    else
        s = zero(T)
        for i in eachindex(x, α, θ)
            ai, ti, xi = convert(T, α[i]), convert(T, θ[i]), convert(T, x[i])
            lt = log(ti)
            value = muladd(ai - one(T), log(abs(xi)), -(xi / ti)) - loggamma(ai) - ai * lt
            value = ifelse((xi == zero(T)) & (ai == one(T)), -lt, value)
            s += ifelse(isfinite(xi) & (xi >= zero(T)), value, convert(T, -Inf))
        end
        s
    end
end
Plaice.scalar_to_scalar_bijector(::BatchedGamma) = Plaice.Log(0.0, 1)
