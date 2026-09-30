"""
    HalfTDist(ν, μ, σ)
    HalfTDist(ν, μ)
    HalfTDist(ν)

Student's t distribution with `ν` degrees of freedom, location `μ` (default 0) and scale
`σ` (default 1), truncated to `[0, ∞)`.

`σ` is a scale parameter, not a standard deviation.

The log normalising constant, including the log probability that the untruncated
distribution assigns to `[0, ∞)`, is computed on construction and stored in the struct.
That probability needs a regularized incomplete beta integral, so the saving is largest
when one `HalfTDist` scores many values.

Sampling inverts the distribution function with a Newton solve, so a draw takes one
uniform noise value whatever `μ/σ` is.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct HalfTDist{V<:Number,M<:Number,S<:Number,B<:Number,P<:Number,L<:Number} <:
       PositiveUnivariateDist
    ν::V
    μ::M
    σ::S
    logbeta_half::B
    logmass::P
    lognorm::L
    function HalfTDist(ν::Number, μ::Number, σ::Number)
        logbeta_half = tdist_logbeta(ν)
        logmass = tdistlogcdf(ν, μ / σ, logbeta_half)
        lognorm = log(σ) + log(ν) / 2 + logbeta_half + logmass
        fields = (ν, μ, σ, logbeta_half, logmass, lognorm)
        return new{map(typeof, fields)...}(fields...)
    end
end

HalfTDist(ν::Number, μ::Number) = HalfTDist(ν, μ, one(μ))
HalfTDist(ν::Number) = HalfTDist(ν, zero(ν), one(ν))

# `log B(1/2, ν/2)`; the t density's normalising constant is `√ν B(1/2, ν/2)`.
@inline tdist_logbeta(ν) = logπ / 2 + loggamma(ν / 2) - loggamma((ν + 1) / 2)

#=
  With `x = ν / (ν + t²)` and `y = t² / (ν + t²)`, the t distribution function is

      F(t) = 1/2 + sign(t) I_y(1/2, ν/2) / 2   near the centre,
      F(t) = I_x(ν/2, 1/2) / 2                 in the lower tail, mirrored for the upper,

  and each form puts the continued fraction on its fast side of the same crossover. In the
  central form the prefactor's `sign(t) √y` is `t / √(ν + t²)`, smooth through `t = 0`, so
  derivatives there are correct. The tail form is kept in log space so a tiny lower tail
  keeps its digits.
=#
function tdistlogcdf(ν::Number, t::Number, logbeta_half::Number)
    fixedlength(ν, t) && return tdistlogcdf_fixed(ν, t, logbeta_half)
    x, y = tdist_xy(ν, t)
    return if y < betainc_crossover(one(ν) / 2, ν / 2)
        tdistlogcdf_centre(ν, t, x, logbeta_half, betacf(one(ν) / 2, ν / 2, y))
    else
        tdistlogcdf_tail(ν, t, x, y, logbeta_half, betacf(ν / 2, one(ν) / 2, x))
    end
end

#=
  Each form is evaluated with its arguments clamped to its own side of the crossover, so
  the form `select` discards stays finite. The central form clamps `t` itself, at `tc`
  where `y = yc`, since it uses `t` and `x` together.
=#
function tdistlogcdf_fixed(ν::Number, t::Number, logbeta_half::Number)
    yc = betainc_crossover(one(ν) / 2, ν / 2)
    xc = one(yc) - yc
    tc = sqrt(ν * yc / xc)
    tl = min(max(t, -tc), tc)
    xl, yl = tdist_xy(ν, tl)
    centre =
        tdistlogcdf_centre(ν, tl, xl, logbeta_half, betacf_fixed(one(ν) / 2, ν / 2, yl))
    x, y = tdist_xy(ν, t)
    xu, yu = min(x, xc), max(y, yc)
    tail = tdistlogcdf_tail(ν, t, xu, yu, logbeta_half, betacf_fixed(ν / 2, one(ν) / 2, xu))
    return select(y < yc, () -> centre, () -> tail)
end

@inline function tdist_xy(ν, t)
    t2 = t^2
    return ν / (ν + t2), t2 / (ν + t2)
end

@inline function tdistlogcdf_centre(ν, t, x, logbeta_half, cf)
    signed_sqrt_y = t / sqrt(ν + t^2)
    return log1p(2 * signed_sqrt_y * exp(ν / 2 * log(x) - logbeta_half) * cf) - logtwo
end

@inline function tdistlogcdf_tail(ν, t, x, y, logbeta_half, cf)
    logtail = ν / 2 * log(x) + log(y) / 2 - logbeta_half - log(ν / 2) + log(cf) - logtwo
    return select(t < zero(t), () -> logtail, () -> log1mexp(logtail))
end

@inline zval(d::HalfTDist, x::Number) = (x - d.μ) / d.σ

@inline function DensityInterface.logdensityof(d::HalfTDist, x::Number)
    ν = d.ν
    value = -(ν + 1) / 2 * log1p(zval(d, x)^2 / ν) - d.lognorm
    return select(x >= zero(x), () -> value, () -> oftype(value, -Inf))
end

const TDISTQUANTILE_MAXITER = 100
const TDISTQUANTILE_STEPS = 8

#=
  The `t ≤ 0` with `log F(t) = logp`, for `logp ≤ -log 2`.

  Newton runs on `w = asinh(t / √ν)`, in which `log F` is concave and close to linear
  both near the centre and in the power-law tail. From the normal start below, `Float64`
  settles within 6 steps for `ν` from `1e-2` to `1e4` and `logp` down to `-700`. Where
  `|t|` exceeds about `1e154`, `t²` overflows inside `tdistlogcdf` and the result is `NaN`.
=#
function tdistlowerquantile(ν::Number, logp::Number, logbeta_half::Number)
    fixedlength(ν, logp) && return tdistlowerquantile_fixed(ν, logp, logbeta_half)
    w = tdistquantile_start(ν, logp)
    tol = sqrt(eps(basefloat(typeof(logp))))
    for _ in 1:TDISTQUANTILE_MAXITER
        step = tdistquantile_step(ν, w, logp, logbeta_half)
        w = min(w - step, zero(w))
        abs(step) <= tol * max(one(w), abs(w)) && break
    end
    # As in `gammalogquantile`, one more step brings the derivative that automatic
    # differentiation carries through the iteration to the accuracy of the value.
    w = min(w - tdistquantile_step(ν, w, logp, logbeta_half), zero(w))
    return sqrt(ν) * sinh(w)
end

function tdistlowerquantile_fixed(ν::Number, logp::Number, logbeta_half::Number)
    w = tdistquantile_start(ν, logp)
    for _ in 1:TDISTQUANTILE_STEPS
        w = min(w - tdistquantile_step(ν, w, logp, logbeta_half), zero(w))
    end
    return sqrt(ν) * sinh(w)
end

@inline logcosh(w) = abs(w) + log1p(exp(-2 * abs(w))) - logtwo

# With `t = √ν sinh(w)`, `d log F / dw = exp(-ν log cosh(w) - log B(1/2, ν/2) - log F)`.
@inline function tdistquantile_step(ν, w, logp, logbeta_half)
    logF = tdistlogcdf(ν, sqrt(ν) * sinh(w), logbeta_half)
    return (logF - logp) * exp(logF + logbeta_half + ν * logcosh(w))
end

# The normal quantile with the first Cornish-Fisher correction for the t distribution.
@inline function tdistquantile_start(ν, logp)
    z = -(sqrt2 * erfcinvt(2 * exp(logp)))
    t = z + (z^3 + z) / (4 * ν)
    return min(asinh(t / sqrt(ν)), zero(t))
end

#=
  Inverse-CDF draw through the upper tail. For a unit draw `u`, the standardised draw
  `r = (x - μ)/σ` solves `P(T > r) = u m`, where `m = F(μ/σ)` is the mass above zero. By
  symmetry `P(T > r) = F(-r)`, so `-r` is the lower quantile of `u m` when that is at most
  one half, and the lower quantile of `1 - u m` is `r` otherwise. Rounding can land the
  draw a hair below zero, so it is clamped there.
=#
function _halft_upper_quantile(d::HalfTDist, u::Number)
    ν, μ, σ, logbeta_half = d.ν, d.μ, d.σ, d.logbeta_half
    logp = log(u) + d.logmass
    lower = logp <= -logtwo
    target = select(lower, () -> logp, () -> log1mexp(logp))
    t = tdistlowerquantile(ν, target, logbeta_half)
    x = muladd(σ, select(lower, () -> -t, () -> t), μ)
    return max(x, zero(x))
end

@inline function Base.rand(rng::AbstractRNG, d::HalfTDist)
    return _halft_upper_quantile(d, one(noisetype(d)) - rand(rng, noisetype(d)))
end
