"""A description of the representation stored in a [`TransformedValue`](@ref)."""
abstract type AbstractTransform end

"""Indicate that a value is an unconstrained, vectorised value."""
struct AsUnconstrainedVec <: AbstractTransform end

"""Indicate that a value is an original-space, vectorised value."""
struct AsVec <: AbstractTransform end

"""Indicate that a value is in its original model-space representation."""
struct AsOriginal <: AbstractTransform end

"""
    FixedTransform(transform)

Store a fixed inverse transform that maps an internally stored value to model space.
"""
struct FixedTransform{F} <: AbstractTransform
    transform::F
end
Base.:(==)(a::FixedTransform, b::FixedTransform) = a.transform == b.transform
Base.isequal(a::FixedTransform, b::FixedTransform) = isequal(a.transform, b.transform)

"""
    TransformedValue(value, transform)

A value together with a description of its current representation. The stored `transform`
describes how to recover the raw value used inside the model.
"""
struct TransformedValue{V,T<:AbstractTransform}
    value::V
    transform::T
end
Base.:(==)(a::TransformedValue, b::TransformedValue) =
    a.value == b.value && a.transform == b.transform
Base.isequal(a::TransformedValue, b::TransformedValue) =
    isequal(a.value, b.value) && isequal(a.transform, b.transform)

"""Return the internally stored value of `tval`."""
get_internal_value(tval::TransformedValue) = tval.value

"""Return the representation descriptor stored in `tval`."""
get_transform(tval::TransformedValue) = tval.transform

"""The coordinate space in which latent parameters and their density are represented."""
abstract type AbstractParameterSpace end

"""
    target_transform(parameter_space, vn) -> AbstractTransform

Return the target representation selected by `parameter_space` for latent variable `vn`.
"""
function target_transform end

"""Select an unconstrained vector representation for every latent variable."""
struct UnconstrainedSpace <: AbstractParameterSpace end
target_transform(::UnconstrainedSpace, ::VarName) = AsUnconstrainedVec()

"""Keep every latent variable in original space."""
struct OriginalSpace <: AbstractParameterSpace end
target_transform(::OriginalSpace, ::VarName) = AsVec()

"""Return the inverse transform from an unconstrained vector to model space."""
function from_unconstrained_vec end

"""Return the inverse vectorisation transform from an original-space vector."""
function from_vec end

"""
    apply_parameter_space(parameter_space, tval, vn, dist)

Interpret the value supplied by a latent source in `parameter_space`. Return the raw
model-space value, its target [`TransformedValue`](@ref), and the forward log absolute
determinant of the Jacobian from model space to the target representation.
"""
function apply_parameter_space(
    parameter_space::AbstractParameterSpace,
    tval::TransformedValue{<:AbstractVector{<:Real},AsUnconstrainedVec},
    vn::VarName,
    dist,
)
    inverse_link = from_unconstrained_vec(dist)
    raw_value, inverse_logjac =
        ChangesOfVariables.with_logabsdet_jacobian(inverse_link, tval.value)
    target = target_transform(parameter_space, vn)
    return if target isa AsUnconstrainedVec
        raw_value, tval, -inverse_logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value, TransformedValue(raw_value, AsOriginal()), zero(LogProbType)
    elseif target isa FixedTransform
        forward = inverse(target.transform)
        value, logjac = ChangesOfVariables.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, target), logjac
    else
        error("unknown target transform: $target")
    end
end

function apply_parameter_space(
    parameter_space::AbstractParameterSpace,
    tval::TransformedValue{<:AbstractVector{<:Real},AsVec},
    vn::VarName,
    dist,
)
    raw_value = from_vec(dist)(tval.value)
    target = target_transform(parameter_space, vn)
    return if target isa AsUnconstrainedVec
        forward = inverse(from_unconstrained_vec(dist))
        value, logjac = ChangesOfVariables.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, AsUnconstrainedVec()), logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value, tval, zero(LogProbType)
    elseif target isa FixedTransform
        forward = inverse(target.transform)
        value, logjac = ChangesOfVariables.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, target), logjac
    else
        error("unknown target transform: $target")
    end
end

function apply_parameter_space(
    parameter_space::AbstractParameterSpace,
    tval::TransformedValue{<:Any,AsOriginal},
    vn::VarName,
    dist,
)
    raw_value = tval.value
    target = target_transform(parameter_space, vn)
    return if target isa AsUnconstrainedVec
        forward = inverse(from_unconstrained_vec(dist))
        value, logjac = ChangesOfVariables.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, AsUnconstrainedVec()), logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value, tval, zero(LogProbType)
    elseif target isa FixedTransform
        forward = inverse(target.transform)
        value, logjac = ChangesOfVariables.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, target), logjac
    else
        error("unknown target transform: $target")
    end
end

function apply_parameter_space(
    parameter_space::AbstractParameterSpace,
    tval::TransformedValue{<:Any,<:FixedTransform},
    vn::VarName,
    dist,
)
    target = target_transform(parameter_space, vn)
    return if target isa AsUnconstrainedVec
        raw_value = tval.transform.transform(tval.value)
        forward = inverse(from_unconstrained_vec(dist))
        value, logjac = ChangesOfVariables.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, AsUnconstrainedVec()), logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value = tval.transform.transform(tval.value)
        raw_value, TransformedValue(raw_value, AsOriginal()), zero(LogProbType)
    elseif target isa FixedTransform
        target == tval.transform ||
            throw(ArgumentError("latent variable $vn has an incompatible fixed transform"))
        raw_value, inverse_logjac = ChangesOfVariables.with_logabsdet_jacobian(
            tval.transform.transform, tval.value
        )
        raw_value, tval, -inverse_logjac
    else
        error("unknown target transform: $target")
    end
end
