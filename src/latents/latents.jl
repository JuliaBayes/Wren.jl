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
`VarNamedTuple` mapping `sitename => RangeAndTransform`.

The fast path is only hit for sampling sites that are encountered once and only once during
model evaluation. This allows us to completely elide VarNamedTuple lookups for these sites.
Usually the VNT lookup is not a problem, but VarNames like `x[1:100000]` can be very slow
due to PartialArray and ArrayLikeBlock checks.

## Submodel handling

The first VNT handles `VarName`s exactly as they appear in the lhs of tilde-statements. Note
that sitenames of submodels are prefixed by the **`VarName`** of the submodel site, rather
than the **sitename** of the submodel site. For example, if we have

```julia
@wren function inner()
    @sample name=:norm x ~ Normal()
end
@wren function outer()
    @sample name=:myinner a ~ inner()
end
```

then the first VNT stores `@vn(a.x) => RangeAndTransform(...)`, and the second VNT stores
`@vn(a.norm) => RangeAndTransform(...)`, **not** `@vn(a.myinner.norm)`. This design allows
us to:

- avoid passing the submodel site name into functions such as `scope`, which would pollute
  the signature with something that is only relevant for Wren (i.e., making it harder to
  use submodels from other sources), and

- handle cases where a submodel contains a slow variable (e.g. `x[1:10000]`) but is sampled
  in a loop. For example

  ```julia
  @wren function inner()
      x = zeros(N)
      x[1:N] ~ dist
  end
  @wren function outer()
      a = Vector{Any}(undef, M)
      for i in 1:M
          a[i] ~ inner()
      end
  end
  ```

  Prefixing with the submodel site would cause all the submodels' `x` variables to have the
  same prefixed sampling site, whereas prefixing with the VarName allows each submodel's `x`
  to have a different entry.

The main inner constructor takes a `VarNamedTuple` and a [`SiteNameTracker`](@ref). This
guarantees that the actual information stored in both fields are consistent.

"""
struct RangesAndTransforms{N<:VarNamedTuple,V<:VarNamedTuple}
    sitename_lookup::N
    slow_vnt::V
    # Derives the fast-path lookup from the sites tracked during model evaluation.
    function RangesAndTransforms(tracker::SiteNameTracker, slow_vnt::VarNamedTuple)
        lookup = sitename_lookup(tracker, slow_vnt)
        return new{typeof(lookup),typeof(slow_vnt)}(lookup, slow_vnt)
    end
    # Used by `scope`, where both fields have already been extracted from a parent
    # RangesAndTransforms and are therefore already consistent with one another.
    function RangesAndTransforms(lookup::VarNamedTuple, slow_vnt::VarNamedTuple)
        return new{typeof(lookup),typeof(slow_vnt)}(lookup, slow_vnt)
    end
end
Base.:(==)(a::RangesAndTransforms, b::RangesAndTransforms) = a.slow_vnt == b.slow_vnt

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
    rat::RangesAndTransforms{<:VarNamedTuple{names,types}},
    vn::VarName,
    ::Val{sitename},
) where {names,types,sitename}
    idx = findfirst(==(sitename), names)
    # The fieldtype check keeps us on the slow path for sitenames whose entry is not
    # actually a RangeAndTransform, i.e. blacklisted sites, plus the pathological case where
    # a plain sampling site shares its name with a submodel prefix.
    return if idx !== nothing && fieldtype(types, idx) <: RangeAndTransform
        :(rat.sitename_lookup.data.$sitename)
    else
        :(rat.slow_vnt[vn])
    end
end

"""
    get_latent(source::FromVector, ...)

When evaluating a model with `FromVector`, Wren first checks the `sitename_lookup` to see if
the `RangeAndTransform` has been stored there. If it hasn't, then it will do the potentially
slower `VarNamedTuple` lookup.

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
function get_latent(source::FromVector, vn::VarName, right, sitename::Val)
    rt = _get_range_and_transform(source.ranges_and_transforms, vn, sitename)
    range, tfm = rt.range, rt.transform
    return TransformedValue(view(source.params, range), tfm)
end

_maybe_get_vnt(source::VarNamedTuple, vn::VarName) =
    haskey(source, vn) ? source[vn] : VarNamedTuple()
function scope(source::FromVector, vn::VarName)
    rat = source.ranges_and_transforms
    # As explained in the constructor docstring for RangesAndTransforms, both VNTs are
    # prefixed using `vn` rather than the sitename, so we can directly index into it.
    return FromVector(
        source.params,
        RangesAndTransforms(rat.sitename_lookup[vn], rat.slow_vnt[vn]),
    )
end
