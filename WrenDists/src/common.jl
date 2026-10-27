# A minimal abstract type for dispatch (SamplerTrivial, DensityKind, etc.)
abstract type WrenDist end

DensityInterface.DensityKind(::WrenDist) = DensityInterface.HasDensity()

function Random.rand(rng::AbstractRNG, sp::Random.SamplerTrivial{<:WrenDist})
    return rand(rng, sp[])
end

Base.rand(d::WrenDist) = rand(Random.default_rng(), d)

# --- Shared utilities ---

@inline select(cond::Bool, iftrue, iffalse) = cond ? iftrue() : iffalse()

# Differentiating logbeta's recursive implementation allocates a tape per sample.
_logbeta(a, b) = logbeta(a, b)
EnzymeRules.@easy_rule(
    _logbeta(a, b),
    @setup(ψ = digamma(a + b)),
    (digamma(a) - ψ, digamma(b) - ψ),
)

wrappedconditions(::Type) = false


@inline function erfcinvt(y::Number)
    valid = (y >= zero(y)) & (y <= 2 * one(y))
    return select(valid, () -> erfcinv(y), () -> oftype(float(y), NaN))
end

basefloat(::Type{T}) where {T<:AbstractFloat} = T
basefloat(::Type{T}) where {T<:Real} = float(T)
basefloat(::Type{Bool}) = Float64
basefloat(::Type{<:Irrational}) = Float64

@inline function noisetype(d::D) where {D<:WrenDist}
    return basefloat(_promoted_paramtype(D))
end

# AD number types either have no rand/randn method, or they draw a different
# stream from the Float64 bulk fill, so draw plain noise. ForwardDiff and
# ReverseDiff define eps(::Type) as eps(valtype(T)), recursively, so
# typeof(eps(T)) is the plain float. A new AD number type needs the same method.
# See https://github.com/JuliaBayes/Wren.jl/pull/238.
@inline function rngtype(d::WrenDist)
    T = noisetype(d)
    return T <: AbstractFloat ? T : typeof(eps(T))
end

@inline function _promoted_paramtype(::Type{D}) where {D}
    return promote_type(ntuple(i -> eltype(fieldtype(D, i)), Val(fieldcount(D)))...)
end

@inline function masstype(d::D, x::Number) where {D<:WrenDist}
    return float(promote_type(_promoted_paramtype(D), typeof(x)))
end
