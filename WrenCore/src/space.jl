"""
    AbstractTransform

A specification of how to transform a variable's value back into its original space. Note
that this might not necessarily contain the actual transformation function: in particular,
[`AsUnconstrainedVec`](@ref) and [`AsVec`](@ref) are just markers that indicate that the
transform function should be derived from the distribution.

Wren provides a few built-in transform types; it is unlikely that you will need to define
your own (if you do, please get in touch and let us know!)

- [`AsUnconstrainedVec`](@ref)
- [`AsVec`](@ref)
- [`AsOriginal`](@ref)
- [`FixedTransform`](@ref)
"""
abstract type AbstractTransform end

"""
    AsUnconstrainedVec <: AbstractTransform

Indicate that a value is an unconstrained, vectorised value. The appropriate transformation
function is `Plaice.from_unconstrained_vec(dist)`, where `dist` is the prior distribution of
the variable.
"""
struct AsUnconstrainedVec <: AbstractTransform end

"""
    AsVec <: AbstractTransform

Indicate that a value is a vectorised value. The appropriate transformation function is
`Plaice.from_vec(dist)`, where `dist` is the prior distribution of the variable.
"""
struct AsVec <: AbstractTransform end

"""
    AsOriginal <: AbstractTransform

Indicates a value that is already in its original space. No transformation is necessary.
"""
struct AsOriginal <: AbstractTransform end

"""
    FixedTransform(transform) <: AbstractTransform

Stores a fixed inverse transform that maps an internally stored value to original space.
"""
struct FixedTransform{F} <: AbstractTransform
    transform::F
end
Base.:(==)(a::FixedTransform, b::FixedTransform) = a.transform == b.transform
Base.isequal(a::FixedTransform, b::FixedTransform) = isequal(a.transform, b.transform)

"""
    TransformedValue(value, transform::AbstractTransform)

A transformed value together with a description of its current representation. The stored
`transform` describes how to recover the untransformed value (in its original space).
"""
struct TransformedValue{V,T<:AbstractTransform}
    value::V
    transform::T
end
Base.:(==)(a::TransformedValue, b::TransformedValue) =
    a.value == b.value && a.transform == b.transform
Base.isequal(a::TransformedValue, b::TransformedValue) =
    isequal(a.value, b.value) && isequal(a.transform, b.transform)

"""
    get_internal_value(tval::TransformedValue) 

Return the transformed value inside `tval`.
"""
get_internal_value(tval::TransformedValue) = tval.value

"""
    get_transform(tval::TransformedValue) -> AbstractTransform

Return the transform that describes how to recover the original value from `tval`.
"""
get_transform(tval::TransformedValue) = tval.transform

"""
    AbstractParameterSpace

The coordinate space in which latent parameters and their density are represented.

An `AbstractParameterSpace` can be thought of as a collection of `AbstractTransform`s, one
for each latent variable in a model. However, while `AbstractTransform`s represent
information that has already been decided (i.e., it is irrevocably tied to the value stored
in a `TransformedValue`), an `AbstractParameterSpace` more resembles an upfront
specification of how one *wants* to evaluate a model.

When evaluating a model with a given `AbstractParameterSpace`, the model's latent variables
will be transformed into the representation that it specifies (using `target_transform`).
The log-Jacobian that is accumulated, and hence the log-density, will be with respect to the
coordinate space specified by the `AbstractParameterSpace`.

There are two built-in `AbstractParameterSpace`s:

- [`UnconstrainedSpace`](@ref): all latent variables are represented in unconstrained,
  vectorised space. This is the default for Wren's vectorised models, particularly because
  this is most suitable for gradient-based inference methods and automatic differentiation.

- [`OriginalSpace`](@ref): all latent variables are represented in their original space.

This can be extended in future to support [`FixedTransform`](@ref)s.
"""
abstract type AbstractParameterSpace end

"""
    target_transform(
        parameter_space::AbstractParameterSpace,
        vn::VarName
    ) -> AbstractTransform

Return the transform that `parameter_space` demands for the latent variable `vn`.
"""
function target_transform end

"""
    UnconstrainedSpace <: AbstractParameterSpace

Evaluate the model in a space where all latent variables are represented as unconstrained
vectors.
"""
struct UnconstrainedSpace <: AbstractParameterSpace end
target_transform(::UnconstrainedSpace, ::VarName) = AsUnconstrainedVec()

"""
    OriginalSpace <: AbstractParameterSpace

Keep every latent variable in original space.
"""
struct OriginalSpace <: AbstractParameterSpace end
target_transform(::OriginalSpace, ::VarName) = AsVec()

# Copy a latent value that the model may write into (see `get_mutated_syms`), since it may
# belong to the caller. The second argument is the latent source (`nothing` for conditioned
# values).
_unalias_latent(::Val{false}, source, value) = value
_unalias_latent(::Val{true}, source, value) = _dealias(value)

"""
    apply_parameter_space(
        parameter_space::AbstractParameterSpace,
        tval::TransformedValue,
        vn::VarName,
        dist,
    )

Convert a latent value into the representation specified by `parameter_space`.

Given a `parameter_space` which says how a latent variable should be represented, and a
[`TransformedValue`](@ref) which contains a value and its current representation (derived
from the [`AbstractLatentSource`](@ref)), returns a tuple of:

- the value in original model space,
- the value in the target representation (a `TransformedValue`), and
- the forward log absolute determinant of the Jacobian from original space to the target
  representation.

Note that when evaluating a model, even if the values are already supplied in unconstrained
space and the target representation is also unconstrained, we still have to perform the
inverse transform to original space for two reasons:

- Calculation of the log-Jacobian.
- The left-hand side of the tilde-statement contains a variable, and we must assign the
  value in original space to that variable.
"""
function apply_parameter_space(
    parameter_space::AbstractParameterSpace,
    tval::TransformedValue{<:AbstractVector{<:Number},AsUnconstrainedVec},
    vn::VarName,
    dist,
)
    inverse_link = Plaice.from_unconstrained_vec(dist)
    raw_value, inverse_logjac = Plaice.with_logabsdet_jacobian(inverse_link, tval.value)
    target = target_transform(parameter_space, vn)
    return if target isa AsUnconstrainedVec
        raw_value, tval, -inverse_logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value, TransformedValue(raw_value, AsOriginal()), zero(LogProbType)
    elseif target isa FixedTransform
        forward = Plaice.inverse(target.transform)
        value, logjac = Plaice.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, target), logjac
    else
        error("unknown target transform: $target")
    end
end
function apply_parameter_space(
    parameter_space::AbstractParameterSpace,
    tval::TransformedValue{<:AbstractVector{<:Number},AsVec},
    vn::VarName,
    dist,
)
    raw_value = Plaice.from_vec(dist)(tval.value)
    target = target_transform(parameter_space, vn)
    return if target isa AsUnconstrainedVec
        forward = Plaice.inverse(Plaice.from_unconstrained_vec(dist))
        value, logjac = Plaice.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, AsUnconstrainedVec()), logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value, tval, zero(LogProbType)
    elseif target isa FixedTransform
        forward = Plaice.inverse(target.transform)
        value, logjac = Plaice.with_logabsdet_jacobian(forward, raw_value)
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
        forward = Plaice.inverse(Plaice.from_unconstrained_vec(dist))
        value, logjac = Plaice.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, AsUnconstrainedVec()), logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value, tval, zero(LogProbType)
    elseif target isa FixedTransform
        forward = Plaice.inverse(target.transform)
        value, logjac = Plaice.with_logabsdet_jacobian(forward, raw_value)
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
        forward = Plaice.inverse(Plaice.from_unconstrained_vec(dist))
        value, logjac = Plaice.with_logabsdet_jacobian(forward, raw_value)
        raw_value, TransformedValue(value, AsUnconstrainedVec()), logjac
    elseif target isa AsVec || target isa AsOriginal
        raw_value = tval.transform.transform(tval.value)
        raw_value, TransformedValue(raw_value, AsOriginal()), zero(LogProbType)
    elseif target isa FixedTransform
        target == tval.transform ||
            throw(ArgumentError("latent variable $vn has an incompatible fixed transform"))
        raw_value, inverse_logjac =
            Plaice.with_logabsdet_jacobian(tval.transform.transform, tval.value)
        raw_value, tval, -inverse_logjac
    else
        error("unknown target transform: $target")
    end
end

"""
    scope(parameter_space::AbstractParameterSpace, vn::VarName) -> AbstractParameterSpace

Given a parameter space in which a model is being executed, return the parameter space in which
a submodel prefixed by `vn` should be executed.
"""
scope(parameter_space::UnconstrainedSpace, ::VarName) = parameter_space
scope(parameter_space::OriginalSpace, ::VarName) = parameter_space
