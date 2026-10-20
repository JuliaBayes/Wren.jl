struct RangeAndTransform{R<:Base.UnitRange,T<:AbstractTransform}
    range::R
    transform::T
end
Base.:(==)(a::RangeAndTransform, b::RangeAndTransform) =
    a.range == b.range && a.transform == b.transform
Base.isequal(a::RangeAndTransform, b::RangeAndTransform) =
    isequal(a.range, b.range) && isequal(a.transform, b.transform)

"""
    VecInfo

All the information needed to derive a vectorised form of a model.

This struct holds a `VarNamedTuple` mapping `VarName => RangeAndTransform`, together with a
fast-path `VarNamedTuple` mapping `sitename => (key => RangeAndTransform)`. The key is
usually a `VarName`; direct integer-vector leaves store their root, length, and first index.

The fast path is only hit for sampling sites that are encountered once and only once during
model evaluation, provided the site's current `VarName` matches its cached key. This
allows us to elide VarNamedTuple lookups when the site still refers to the same variable.
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
`@vn(a.norm) => (@vn(x) => RangeAndTransform(...))`, **not** `@vn(myinner.norm)`. The
cached name is local to the submodel, so it can be compared without its prefix during
evaluation. This design allows us to:

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

The main inner constructor takes a `VarNamedTuple` and a [`SitenameCountAccumulator`](@ref).
This guarantees that the two maps are consistent.
"""
struct VecInfo{N<:VarNamedTuple,V<:VarNamedTuple}
    sitename_to_rat::N
    varname_to_rat::V
end
function VecInfo(sca::SitenameCountAccumulator, varnames_to_rats::VarNamedTuple)
    sitenames_to_varnames = sca.entries
    # TODO: This means that sitenames_to_rats will actually be a mapping of sitenames to
    # either cached key-RAT pairs or SeenMultipleTimes. Do we want to get rid of the latter?
    sitenames_to_rats = VarNames.map_pairs!!(sitenames_to_varnames) do (site, vn)
        if vn isa SeenMultipleTimes
            return vn
        end
        # Remove the submodel namespace once, rather than on every cache lookup.
        prefix = VarNames.oinit(VarNames.varname_to_optic(site))
        local_vn = if prefix isa VarNames.Iden
            vn
        else
            VarNames.unprefix(vn, VarNames.optic_to_varname(prefix))
        end
        return _cache_varname_key(local_vn) => varnames_to_rats[vn]
    end
    return VecInfo(sitenames_to_rats, varnames_to_rats)
end
Base.:(==)(a::VecInfo, b::VecInfo) = a.varname_to_rat == b.varname_to_rat

"""
    FromVector(
        params::AbstractVector,
        vecinfo::VecInfo
    ) <: AbstractLatentSource

Provide latent values from a vector of flattened parameters, together with a
`VecInfo` that provides information on how to recover the values in their
original space.
"""
struct FromVector{V<:AbstractVector,R<:VecInfo} <: AbstractLatentSource
    params::V
    vecinfo::R
end

_cache_varname_key(vn::VarName) = deepcopy(vn)

function _cache_varname_key(
    vn::VarName{
        sym,
        VarNames.Index{Tuple{Vector{Int}},NamedTuple{(),Tuple{}},VarNames.Iden},
    },
) where {sym}
    indices = only(VarNames.getoptic(vn).ix)
    # A valid leaf lookup selects a complete disjoint block, so one shared index
    # identifies it. Snapshot the key because the index vector can be mutated.
    return isempty(indices) ? deepcopy(vn) : (Val(sym), length(indices), first(indices))
end

_cache_varname_matches(a::VarName, b::VarName) = isequal(a, b)
_cache_varname_matches(::VarName, ::Tuple{Val{sym},Int,Int}) where {sym} = false

function _cache_varname_matches(
    vn::VarName{
        sym,
        VarNames.Index{Tuple{Vector{Int}},NamedTuple{(),Tuple{}},VarNames.Iden},
    },
    key::Tuple{Val{sym},Int,Int},
) where {sym}
    indices = only(VarNames.getoptic(vn).ix)
    return length(indices) == key[2] && !isempty(indices) && first(indices) == key[3]
end

function _cache_varname_matches(
    a::VarName{sym,<:VarNames.Index{Tuple{Vector{Int}}}},
    b::VarName{sym,<:VarNames.Index{Tuple{Vector{Int}}}},
) where {sym}
    ao, bo = VarNames.getoptic(a), VarNames.getoptic(b)
    ai, bi = only(ao.ix), only(bo.ix)
    # Integer vectors have identical == and isequal semantics, but == uses memcmp.
    return (ai === bi || ai == bi) && isequal(ao.kw, bo.kw) && isequal(ao.child, bo.child)
end

@generated function _get_range_and_transform(
    vi::VecInfo{<:VarNamedTuple{names,types}},
    vn::VarName{sym},
    ::Val{sitename},
) where {names,types,sym,sitename}
    idx = findfirst(==(sitename), names)
    # Need to check fieldtype because vi.sitename_to_rat might actually still contain
    # SeenMultipleTimes objects. TODO(penelopeysm) Maybe this is worth fixing?
    return if idx !== nothing &&
              fieldtype(types, idx) <:
              Pair{<:Union{VarName,Tuple{Val{sym},Int,Int}},<:RangeAndTransform}
        quote
            cached = vi.sitename_to_rat.data.$sitename
            if _cache_varname_matches(vn, cached.first)
                cached.second
            else
                vi.varname_to_rat[vn]
            end
        end
    else
        :(vi.varname_to_rat[vn])
    end
end

"""
    get_latent(source::FromVector, ...)

When evaluating a model with `FromVector`, Wren checks `sitename_to_rat` for a matching
key. Otherwise it uses the potentially slower `VarNamedTuple` lookup.

The implicit assumption here is that if `sitename` was only encountered once during the
construction of the `VecInfo`, then it will only be encountered once during
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
    rt = _get_range_and_transform(source.vecinfo, vn, sitename)
    range, tfm = rt.range, rt.transform
    return TransformedValue(view(source.params, range), tfm)
end

function scope(source::FromVector, vn::VarName)
    vi = source.vecinfo
    # As explained in the constructor docstring for VecInfo, both VNTs are prefixed using
    # `vn` rather than the sitename, so we can directly index into it.
    #
    # Note that this implementation **forces** vi.sitename_to_rat to always contain an
    # entry for `vn`, even if the submodel doesn't contain any unique sampling sites.
    # For example, consider:
    #
    # @wren function inner()
    #     x = zeros(10)
    #     for i in 1:10
    #         x[i] ~ Normal()
    #     end
    # end
    #
    # @wren function outer()
    #     a ~ inner()
    # end
    #
    # When running the submodel, there are no unique sampling sites, so in principle we
    # don't need to record anything in the sitename_to_rat VNT. However, we can't completely
    # elide it: we need to retain an empty VNT under the key `@vn(a)` so that the getindex
    # call `vi.sitename_to_rat[vn]` doesn't throw an error. This creates an invariant that
    # must be upheld during *construction* of `sitename_to_rat`. Specifically, see
    # `combine(::SitenameCountAccumulator, ...)` for where this is fulfilled.
    #
    # It would be 'simpler' to just check if `haskey(vi.sitename_to_rat, vn)` and return an
    # empty VNT if not. However, this is not type stable. Because scope(::FromVector, ...)
    # is called during the hot path of model evaluation, we can't afford to introduce type
    # instabilities.
    return FromVector(source.params, VecInfo(vi.sitename_to_rat[vn], vi.varname_to_rat[vn]))
end
