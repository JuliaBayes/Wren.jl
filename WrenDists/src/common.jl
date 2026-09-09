# A minimal abstract type for dispatch (SamplerTrivial, DensityKind, etc.)
abstract type WrenDist end

DensityInterface.DensityKind(::WrenDist) = DensityInterface.HasDensity()

function Random.rand(rng::AbstractRNG, sp::Random.SamplerTrivial{<:WrenDist})
    return rand(rng, sp[])
end

Base.rand(d::WrenDist) = rand(Random.default_rng(), d)

# --- Shared utilities ---

@inline select(cond::Bool, iftrue, iffalse) = cond ? iftrue() : iffalse()

wrappedconditions(::Type) = false

@inline function logt(x::Number)
    return select(x < zero(x), () -> oftype(float(x), NaN), () -> log(x))
end

@inline function log1pt(x::Number)
    return select(x < -one(x), () -> oftype(float(x), NaN), () -> log1p(x))
end

@inline function log1mexpt(a::Number)
    return select(a > -logtwo, () -> logt(-expm1(a)), () -> log1p(-exp(a)))
end

@inline function xlogyt(x::Number, y::Number)
    ly = logt(y)
    return select((x == zero(x)) & !isfinite(ly), () -> zero(float(x * ly)), () -> x * ly)
end

@inline function logbetat(α::Number, β::Number)
    valid = (α > zero(α)) & (β > zero(β))
    a = select(valid, () -> α, () -> one(α))
    b = select(valid, () -> β, () -> one(β))
    v = loggamma(a) + loggamma(b) - loggamma(a + b)
    return select(valid, () -> v, () -> oftype(v, NaN))
end

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

@inline function _promoted_paramtype(::Type{D}) where {D}
    return promote_type(ntuple(i -> eltype(fieldtype(D, i)), Val(fieldcount(D)))...)
end

@inline function masstype(d::D, x::Number) where {D<:WrenDist}
    return float(promote_type(_promoted_paramtype(D), typeof(x)))
end
