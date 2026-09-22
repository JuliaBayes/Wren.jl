module WrenDistsReactantExt

using WrenDists: WrenDists, GAMMAINC_TERMS, GAMMAQUANTILE_STEPS, TDISTQUANTILE_STEPS
using WrenDists: basefloat, floortiny, select
using WrenDists: gammaquantile_start, gammaquantile_step
using WrenDists: tdistquantile_start, tdistquantile_step
using Reactant: Reactant, TracedRNumber, TracedRArray, @trace
using SpecialFunctions: SpecialFunctions, loggamma

WrenDists.basefloat(::Type{TracedRNumber{T}}) where {T} = WrenDists.basefloat(T)

@inline function WrenDists.select(cond::TracedRNumber{Bool}, iftrue, iffalse)
    return ifelse(cond, iftrue(), iffalse())
end

WrenDists.wrappedconditions(::Type{<:TracedRNumber}) = true

function WrenDists.gammap_series_fixed(a::T, x::T) where {T<:TracedRNumber}
    term = one(T)
    total = one(T)
    @trace for n in 1:GAMMAINC_TERMS
        term = term * (x / (a + n))
        total = total + term
    end
    return muladd(a, log(x), -x) - loggamma(a + one(T)) + log(total)
end

function WrenDists.gammaq_cf_fixed(a::T, x::T) where {T<:TracedRNumber}
    F = basefloat(T)
    tiny = convert(T, sqrt(floatmin(F)))
    b = x + one(T) - a
    c = convert(T, inv(sqrt(floatmin(F))))
    d = inv(b)
    h = inv(b)
    two = 2 * one(T)
    @trace for i in 1:GAMMAINC_TERMS
        an = -i * (i - a)
        b = b + two
        d = inv(floortiny(an * d + b, tiny))
        c = floortiny(b + an / c, tiny)
        h = h * (d * c)
    end
    return muladd(a, log(x), -x) - loggamma(a) + log(h)
end

function WrenDists.gammalogquantile_fixed(a::T, p::T) where {T<:TracedRNumber}
    lower = p <= one(T) / 2
    target = select(lower, () -> log(p), () -> log1p(-p))
    lg = loggamma(a)
    u = gammaquantile_start(a, p)
    @trace for k in 1:GAMMAQUANTILE_STEPS
        u = u - gammaquantile_step(a, lg, u, exp(u), lower, target)
    end
    return u
end

function WrenDists.tdistlowerquantile_fixed(
    ν::Number,
    logp::TracedRNumber,
    logbeta_half::Number,
)
    w = tdistquantile_start(ν, logp)
    @trace for k in 1:TDISTQUANTILE_STEPS
        w = min(w - tdistquantile_step(ν, w, logp, logbeta_half), zero(w))
    end
    return sqrt(ν) * sinh(w)
end

# TODO(penelopeysm) This is piracy!! These were upstreamed in
# https://github.com/EnzymeAD/Reactant.jl/pull/3273 but not yet released.
# Remove this ASAP after the next Reactant release
function SpecialFunctions.erfcinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(one(x) - x)
end
function SpecialFunctions.erfinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(x)
end

end
