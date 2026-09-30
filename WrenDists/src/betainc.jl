#=
  The continued fraction for the regularized incomplete beta integral,

      I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * betacf(a, b, x),

  from Numerical Recipes §6.4, evaluated with the modified Lentz method. It converges
  quickly for `x` below `betainc_crossover(a, b)`; above it, `I_x(a, b) = 1 - I_{1-x}(b, a)`
  puts the fraction on the fast side.

  As in `gammainc.jl`, numbers whose comparisons give a `Bool` run the fraction until it
  settles, and traced numbers run a fixed number of terms.
=#

const BETAINC_MAXITER = 10_000

#=
  The fixed-length path spends this many terms on each fraction whatever the shape. For
  `b = 1/2` or `a = 1/2` on the fast side, the cases the t distribution needs, the
  `Float64` fraction settles within 52 terms for every other shape from `1e-3` to `1e14`.
  Other shapes have not been measured.
=#
const BETAINC_TERMS = 64

@inline betainc_crossover(a, b) = (a + one(a)) / (a + b + 2 * one(a))

function betacf(a::Number, b::Number, x::Number)
    F = basefloat(typeof(x))
    tol = eps(F)
    tiny = oftype(x, sqrt(floatmin(F)))
    c = one(x)
    d = inv(floortiny(one(x) - (a + b) * x / (a + one(a)), tiny))
    h = d
    for m in 1:BETAINC_MAXITER
        even, odd = betacf_coefs(a, b, x, m)
        d = inv(floortiny(one(x) + even * d, tiny))
        c = floortiny(one(x) + even / c, tiny)
        h *= d * c
        d = inv(floortiny(one(x) + odd * d, tiny))
        c = floortiny(one(x) + odd / c, tiny)
        delta = d * c
        h *= delta
        abs(delta - one(delta)) <= tol && break
    end
    return h
end

function betacf_fixed(a::Number, b::Number, x::Number)
    F = basefloat(typeof(x))
    tiny = oftype(x, sqrt(floatmin(F)))
    c = one(x)
    d = inv(floortiny(one(x) - (a + b) * x / (a + one(a)), tiny))
    h = d
    for m in 1:BETAINC_TERMS
        even, odd = betacf_coefs(a, b, x, m)
        d = inv(floortiny(one(x) + even * d, tiny))
        c = floortiny(one(x) + even / c, tiny)
        h *= d * c
        d = inv(floortiny(one(x) + odd * d, tiny))
        c = floortiny(one(x) + odd / c, tiny)
        h *= d * c
    end
    return h
end

@inline function betacf_coefs(a, b, x, m)
    even = m * (b - m) * x / ((a + 2m - 1) * (a + 2m))
    odd = -(a + m) * (a + b + m) * x / ((a + 2m) * (a + 2m + 1))
    return even, odd
end
