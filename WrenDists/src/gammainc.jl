#=
  The regularized incomplete gamma integrals, in log space. Returning logarithms keeps
  a tail readable where the probability itself underflows, and every step stays in the
  type of the arguments, so `BigFloat` keeps its precision.

  There are two paths, chosen by `wrappedconditions`. Numbers whose comparisons give a
  `Bool` run each expansion until its terms stop changing the result. Traced numbers run
  a fixed number of terms with `select` in place of every value-driven branch, so
  Reactant can compile the loop.

  Types come from the values rather than from a shared type parameter, because the
  numbers a reverse-mode tape produces need not share a type with the ones it was given.
=#

#=
  Both expansions converge geometrically once `x` is on its own side of `a + 1`. At the
  crossover they need about `√(2a log(1/ε))` terms to reach the rounding error `ε`:
  `8.5√a` in `Float64`, more at higher precision. This bound covers `Float64` shapes up
  to roughly `10^6`. Past it the remaining terms are dropped and the result loses
  accuracy.
=#
const GAMMAINC_MAXITER = 10_000

#=
  The fixed-length path spends this many terms on each expansion whatever the shape,
  so it is exact only up to `gammainc_maxshape` and returns `NaN` above it.
=#
const GAMMAINC_TERMS = 256

@inline function gammainc_maxshape(::Type{T}) where {T}
    ε = eps(basefloat(T))
    return GAMMAINC_TERMS^2 / (-2 * log(ε))
end

# Either argument may be the traced one.
@inline fixedlength(a, x) = wrappedconditions(typeof(a)) | wrappedconditions(typeof(x))

@inline function loggammap(a::Number, x::Number)
    fixedlength(a, x) && return loggammapq_fixed(a, x)[1]
    return x < a + one(a) ? gammap_series(a, x) : log1mexp(gammaq_cf(a, x))
end

@inline function loggammaq(a::Number, x::Number)
    fixedlength(a, x) && return loggammapq_fixed(a, x)[2]
    return x < a + one(a) ? log1mexp(gammap_series(a, x)) : gammaq_cf(a, x)
end

@inline function loggammapq_fixed(a::Number, x::Number)
    crossover = a + one(a)
    lower = x < crossover
    ls = gammap_series_fixed(a, min(x, crossover))
    lq = gammaq_cf_fixed(a, max(x, crossover))
    logp = select(lower, () -> ls, () -> log1mexp(lq))
    logq = select(lower, () -> log1mexp(ls), () -> lq)
    exact = a <= gammainc_maxshape(typeof(x))
    nan = oftype(ls, NaN)
    return (select(exact, () -> logp, () -> nan), select(exact, () -> logq, () -> nan))
end

function gammap_series(a::Number, x::Number)
    tol = eps(basefloat(typeof(x)))
    term = one(x)
    total = one(x)
    n = 0
    while (n < GAMMAINC_MAXITER) & (abs(term) > tol * abs(total))
        n += 1
        term *= x / (a + n)
        total += term
    end
    return muladd(a, log(x), -x) - loggamma(a + one(a)) + log(total)
end

function gammap_series_fixed(a::Number, x::Number)
    term = one(x)
    total = one(x)
    for n in 1:GAMMAINC_TERMS
        term *= x / (a + n)
        total += term
    end
    return muladd(a, log(x), -x) - loggamma(a + one(a)) + log(total)
end

function gammaq_cf(a::Number, x::Number)
    F = basefloat(typeof(x))
    tol = eps(F)
    tiny = oftype(x, sqrt(floatmin(F)))
    b = x + one(x) - a
    c = oftype(x, inv(sqrt(floatmin(F))))
    d = inv(b)
    h = inv(b)
    zeroed = 0
    for i in 1:GAMMAINC_MAXITER
        an = -i * (i - a)
        (zeroed == 0) & (abs(an) < tiny) && (zeroed = i)
        b += 2 * one(x)
        d = an * d + b
        abs(d) < tiny && (d = tiny)
        c = b + an / c
        abs(c) < tiny && (c = tiny)
        d = inv(d)
        delta = d * c
        h *= delta
        settled = abs(delta - one(delta)) <= tol
        # tol is a plain float even when F is an AD type, so this gives the base precision.
        settled & ((zeroed == 0) | (i >= zeroed + precision(tol))) && break
    end
    return muladd(a, log(x), -x) - loggamma(a) + log(h)
end

function gammaq_cf_fixed(a::Number, x::Number)
    F = basefloat(typeof(x))
    tiny = oftype(x, sqrt(floatmin(F)))
    b = x + one(x) - a
    c = oftype(x, inv(sqrt(floatmin(F))))
    d = inv(b)
    h = inv(b)
    for i in 1:GAMMAINC_TERMS
        an = -i * (i - a)
        b += 2 * one(x)
        d = inv(floortiny(an * d + b, tiny))
        c = floortiny(b + an / c, tiny)
        h *= d * c
    end
    return muladd(a, log(x), -x) - loggamma(a) + log(h)
end

@inline floortiny(v, tiny) = select(abs(v) < tiny, () -> tiny, () -> v)
