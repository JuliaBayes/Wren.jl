"""
    AbstractLatentSource

An abstract type representing somewhere from which the values of latent variables can be
obtained.

Wren provides several subtypes of `AbstractLatentSource`:

- [`FromPrior`](@ref): samples latent variables from their prior distributions
- [`FromParams`](@ref): provides latent variables in a `VarNamedTuple`
- [`FromVector`](@ref): provides a vector of flattened latent variables, together with
  information about how to convert them back into the original space

Custom subtypes of this can be created to customise model execution. To be a valid
`AbstractLatentSource`, a type must implement

- [`get_latent`](@ref)(source, vn, dist) to obtain a latent variable from the source, and
- [`scope`](@ref) to obtain a latent source that applies only to a subset of variables.
  This is used in submodel evaluation: if a model contains `a ~ submodel()`, then the
  latent source used to evaluate `submodel` is obtained by calling `scope(source, @vn(a))`.
"""
abstract type AbstractLatentSource end

"""
    get_latent(source::AbstractLatentSource, vn::VarName, dist) -> TransformedValue

Obtain a value for the latent variable `vn` from `source`. `dist` is the prior distribution
of the variable, which occurs on the right-hand side of the model's tilde-statement.

This function should return a `TransformedValue` which specifies both the value as well as
the way to convert the value back to the model's original space. Note that this allows for
lazy evaluation of the transformation, and thus `get_latent` should be as lazy as possible,
and avoid performing any transformations on the value before returning it.
"""
function get_latent end

"""
    scope(source::AbstractLatentSource, vn::VarName) -> AbstractLatentSource

Restrict `source` such that it only provides values for sub-variables of `vn`. This is used
when evaluating submodels with a prefix `vn`.

This is best explained through an example. Suppose you have a model with

```julia
@model function inner()
    x ~ Normal(0, 1)
end
@model function outer()
    a ~ inner()
end
```

When evaluating `outer`, the latent source may contain a value for the variable `a.x`. For
example, you might have `FromParams(VarNamedTuple(a=VarNamedTuple(x=1.0)))`. However, when
evaluating `inner`, the submodel does not know about the prefix `a`: Wren will call `scope`
to obtain a subsetted latent source which only contains the variable `x`.
"""
function scope end

"""
    FromPrior([rng::AbstractRNG=Random.default_rng()]) <: AbstractLatentSource

Sample latent values from their prior distributions, optionally with a given random number
generator.
"""
struct FromPrior{R<:AbstractRNG} <: AbstractLatentSource
    rng::R
end
FromPrior() = FromPrior(default_rng())
get_latent(source::FromPrior, ::VarName, right) =
    TransformedValue(rand(source.rng, right), AsOriginal())
scope(source::FromPrior, ::VarName) = source

"""
    FromParams(
        params::VarNamedTuple,
        fallback::Union{AbstractLatentSource,Nothing}=nothing
    ) <: AbstractLatentSource

Take latent values from the given `VarNamedTuple`. If the `VarNamedTuple` does not contain a
value for a variable, then the `fallback` source is used, or an error thrown if `fallback`
is `nothing`.
"""
struct FromParams{P<:VarNamedTuple,S<:Union{AbstractLatentSource,Nothing}} <:
       AbstractLatentSource
    params::P
    fallback::S
end
FromParams(params::VarNamedTuple) = FromParams(params, nothing)
function get_latent(source::FromParams, vn::VarName, right)
    return if haskey(source.params, vn)
        value = source.params[vn]
        value isa TransformedValue ? value : TransformedValue(value, AsOriginal())
    else
        source.fallback === nothing && error("No value was provided for variable $vn")
        get_latent(source.fallback, vn, right)
    end
end
function scope(source::FromParams, vn::VarName)
    child_fallback = if source.fallback === nothing
        nothing
    else
        scope(source.fallback, vn)
    end
    return if haskey(source.params, vn)
        child_params = source.params[vn]
        child_params isa VarNamedTuple ||
            throw(ArgumentError("the parameter $vn must contain a VarNamedTuple"))
        FromParams(child_params, child_fallback)
    else
        child_fallback
    end
end

struct RangeAndTransform{R<:Base.UnitRange,T<:AbstractTransform}
    range::R
    transform::T
end
Base.:(==)(a::RangeAndTransform, b::RangeAndTransform) =
    a.range == b.range && a.transform == b.transform
Base.isequal(a::RangeAndTransform, b::RangeAndTransform) =
    isequal(a.range, b.range) && isequal(a.transform, b.transform)

"""
    FromVector(
        params::AbstractVector,
        ranges_and_transforms::VarNamedTuple
    ) <: AbstractLatentSource

Provide latent values from a vector of flattened parameters, together with a `VarNamedTuple`
that provides information on how to recover the values in their original space.

The `ranges_and_transforms` argument should be a `VarNamedTuple` where each value is a
`RangeAndTransform` object. This object specifies:

- a `range`, which indicates the section of the `params` vector where the variable lives;
- a `transform`, which indicates how to convert the value back to the original space.
"""
struct FromVector{V<:AbstractVector,R<:VarNamedTuple} <: AbstractLatentSource
    params::V
    ranges_and_transforms::R
end
_get_range_and_transform(ranges_and_transforms::VarNamedTuple, vn::VarName) =
    ranges_and_transforms[vn]
function get_latent(source::FromVector, vn::VarName, right)
    rt = _get_range_and_transform(source.ranges_and_transforms, vn)
    range, tfm = rt.range, rt.transform
    return TransformedValue(view(source.params, range), tfm)
end
function scope(source::FromVector, vn::VarName)
    return FromVector(source.params, source.ranges_and_transforms[vn])
end
