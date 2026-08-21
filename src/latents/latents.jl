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
    get_latent(
        source::AbstractLatentSource,
        vn::VarName,
        dist,
        ::Val{sitename}
    ) where {sitename} -> TransformedValue

Obtain a value for the latent variable `vn` from `source`.

`dist` is the prior distribution of the variable, which occurs on the right-hand side of the
model's tilde-statement. `sitename` is a compile-time Symbol identifying the sampling site
in the model source code.

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
@wren function inner()
    x ~ Normal(0, 1)
end
@wren function outer()
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
get_latent(source::FromPrior, ::VarName, right, ::Val) =
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
function get_latent(source::FromParams, vn::VarName, right, sitename::Val)
    return if haskey(source.params, vn)
        value = source.params[vn]
        value isa TransformedValue ? value : TransformedValue(value, AsOriginal())
    else
        source.fallback === nothing && error("No value was provided for variable $vn")
        get_latent(source.fallback, vn, right, sitename)
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
    RangesAndTransforms

Holds a `VarNamedTuple` mapping `VarName => RangeAndTransform`, together with a fast-path
`NamedTuple` mapping `sitename => RangeAndTransform`.

The inner constructor takes a `VarNamedTuple` and a mapping of `sitename => VarName` pairs.
This guarantees that the actual information stored in both fields are consistent.

When evaluating a model with `FromVector`, Wren first checks the `sitename_lookup` to see if
the `RangeAndTransform` has been stored there. If it hasn't, then it will do the potentially
slower `VarNamedTuple` lookup.

The fast path is only hit for sampling sites that are encountered once and only once during
model evaluation. This allows us to completely elide VarNamedTuple lookups for these sites.
Usually that's not a problem, but VarNames like `x[1:100000]` can be very slow due to
PartialArray and ArrayLikeBlock checks.

The implicit assumption here is that if `sitename` was only encountered once during the
construction of the `RangesAndTransforms`, then it will only be encountered once during
every model evaluation. This is true of essentially any model that is amenable to
vectorisation.

An interesting case to ponder is a model like this. In the absence of this fast path (i.e.,
if there was only VNT-based lookup like DynamicPPL), this is compatible with vectorisation
as long as `dist1` and `dist2` have the same `Plaice.unconstrained_vec_size`, because the
resulting `RangeAndTransform` will be a constant:

```julia
if something
    x ~ dist1
else
    x ~ dist2
end
```

With the fast path, it's quite possible that the sitename for the `if` branch will be
stored, and the sitename for the `else` branch won't. This means that during model
evaluation, the fast path will only work if we hit the same branch. However, even if we hit
the other branch, the slow path will still work (because `x`'s `RangeAndTransform` will
still be stored in the VNT!)
"""
struct RangesAndTransforms{N<:NamedTuple,V<:VarNamedTuple}
    sitename_lookup::N
    vnt::V
    function RangesAndTransforms(
        vnt::VarNamedTuple,
        sitename_entries::NamedTuple=NamedTuple(),
    )
        sitename_lookup = NamedTuple()
        for (sitename, vn) in pairs(sitename_entries)
            sitename_lookup = merge(sitename_lookup, NamedTuple{(sitename,)}((vnt[vn],)))
        end
        return new{typeof(sitename_lookup),typeof(vnt)}(sitename_lookup, vnt)
    end
end
Base.:(==)(a::RangesAndTransforms, b::RangesAndTransforms) = a.vnt == b.vnt

"""
    FromVector(
        params::AbstractVector,
        ranges_and_transforms::RangesAndTransforms
    ) <: AbstractLatentSource

Provide latent values from a vector of flattened parameters, together with a
`RangesAndTransforms` that provides information on how to recover the values in their
original space.
"""
struct FromVector{V<:AbstractVector,R<:RangesAndTransforms} <: AbstractLatentSource
    params::V
    ranges_and_transforms::R
end
@generated function _get_range_and_transform(
    rat::RangesAndTransforms{<:NamedTuple{names}},
    vn::VarName,
    ::Val{sitename},
) where {names,sitename}
    return if sitename in names
        :(rat.sitename_lookup.$sitename)
    else
        :(rat.vnt[vn])
    end
end
function get_latent(source::FromVector, vn::VarName, right, sitename::Val)
    rt = _get_range_and_transform(source.ranges_and_transforms, vn, sitename)
    range, tfm = rt.range, rt.transform
    return TransformedValue(view(source.params, range), tfm)
end
function scope(source::FromVector, vn::VarName)
    child_vnt = source.ranges_and_transforms.vnt[vn]
    return FromVector(source.params, RangesAndTransforms(child_vnt))
end
