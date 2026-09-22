"""
    HalfTDist(ν, μ, σ)
    HalfTDist(ν)

Student's t distribution with `ν` degrees of freedom, location `μ` (default 0) and scale
`σ` (default 1), truncated to `[0, ∞)`.

The log normalising constant, including the log probability that the untruncated
distribution assigns to `[0, ∞)`, is computed on construction and stored in the struct.
That probability needs a regularized incomplete beta integral, so the saving is largest
when one `HalfTDist` scores many values.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct HalfTDist{V<:Number,M<:Number,S<:Number,L<:Number} <: PositiveUnivariateDist
    ν::V
    μ::M
    σ::S
    lognorm::L
    function HalfTDist(ν::Number, μ::Number, σ::Number)
        logbeta_half = tdist_logbeta(ν)
        lognorm = log(σ) + log(ν) / 2 + logbeta_half + tdistlogcdf(ν, μ / σ, logbeta_half)
        return new{typeof(ν),typeof(μ),typeof(σ),typeof(lognorm)}(ν, μ, σ, lognorm)
    end
end

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

#=
  Rejection from the untruncated distribution, drawn as a normal over the root of a
  `Gamma(ν/2, 2/ν)` draw. The expected number of attempts is the reciprocal of the mass
  above zero, which grows without bound as `μ/σ` falls.
=#
function Base.rand(rng::AbstractRNG, d::HalfTDist)
    ν = d.ν
    while true
        z = randn(rng, noisetype(d))
        w = rand(rng, Gamma(ν / 2, 2 / ν))
        x = muladd(d.σ, z / sqrt(w), d.μ)
        x >= zero(x) && return x
    end
end
