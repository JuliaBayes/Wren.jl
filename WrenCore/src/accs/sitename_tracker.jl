const SITENAMECOUNT_ACCNAME = :SitenameCountAccumulator

"""
    SeenMultipleTimes

Sentinel value that marks a sitename that was hit more than once during model evaluation.
"""
struct SeenMultipleTimes end

"""
    SitenameCountAccumulator

Accumulator that identifies sampling sites hit exactly once during model evaluation.

After evaluating a model, the `entries` field should contain a mapping of *(maybe-prefixed)
sitenames to (maybe-prefixed) `VarName`s*.

For example, take the model

```julia
@wren function model(x)
    @sample name=:sna a ~ Normal()
    for i in eachindex(x)
        @sample name=:snxi x[i] ~ Normal()
    end
end
```

Here, the `sna` site is only hit once during model evaluation, but `snx` is hit more than
once (as long as `x` has length 2 or more). Thus, after evaluating the model with a
`SitenameCountAccumulator`, it should contain the mappings `@vn(sna) => @vn(a)` and `@vn(snx) =>
SeenMultipleTimes()`.

On the first hit of a `sitename`, its `VarName` is stored in a `VarNamedTuple` under the key
`VarName{sitename}()`. On any subsequent hit, the entry is replaced with
[`SeenMultipleTimes`](@ref) so it is never re-added.

## Submodel handling

In the following example:

```julia
@wren function inner()
    @sample name=:sna a ~ Normal()
end

@wren function outer()
    @sample name=:snx x ~ inner()
end
```

After evaluating `outer()`, the `entries` field of the `SitenameCountAccumulator` should contain a
mapping of `@vn(x.sna) => @vn(x.a)`. **Note:** the prefixed sitename is prefixed with the
parent VarName, not the parent sitename (`snx`). The rationale for this is explained in
[`VecInfo`](@ref).
"""
struct SitenameCountAccumulator{V<:VarNamedTuple} <: AbstractAccumulator
    entries::V
end

SitenameCountAccumulator() = SitenameCountAccumulator(VarNamedTuple())

accumulator_name(::Type{<:SitenameCountAccumulator}) = SITENAMECOUNT_ACCNAME

function accumulate_assume!!(
    acc::SitenameCountAccumulator{<:VarNamedTuple{names}},
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    ::Val{sitename},
) where {names,sitename}
    new_entry = sitename in names ? SeenMultipleTimes() : vn
    # No template needed since it's a simple VarName.
    new_entries = VarNames.setindex!!(acc.entries, new_entry, VarName{sitename}())
    return SitenameCountAccumulator(new_entries)
end

function accumulate_observe!!(
    acc::SitenameCountAccumulator,
    val,
    vn,
    right,
    template,
    @nospecialize(sitename::Val),
)
    return acc
end

reset(::SitenameCountAccumulator) = SitenameCountAccumulator()
Base.copy(acc::SitenameCountAccumulator) = acc

function _combine(acc1::SitenameCountAccumulator, acc2::SitenameCountAccumulator)
    # Replace any of acc2's entries that are already present in acc1 with SeenMultipleTimes
    deduped = VarNames.map_pairs!!(acc2.entries) do (key, value)
        haskey(acc1.entries, key) ? SeenMultipleTimes() : value
    end
    # We need to use `merge` here instead of some function that iterates over keys (like
    # map_pairs!!) because empty VNTs are not treated as keys:
    #
    #     v = VarNamedTuple(x = VarNamedTuple())
    #     keys(v) # Returns [].
    #
    # and we want to preserve empty VNTs in order to make sure that a hot path (namely
    # `scope(::FromVector, ...)`) is type stable. See the comments there for more info.
    return SitenameCountAccumulator(merge(acc1.entries, deduped))
end

function combine_from_scope(
    acc1::SitenameCountAccumulator,
    acc2::SitenameCountAccumulator,
    vn::VarName,
    template,
)
    prefixed = VarNames.map_values!!(acc2.entries) do v
        v isa SeenMultipleTimes ? v : VarNames.prefix(v, vn)
    end
    nested = VarNames.templated_setindex!!(VarNamedTuple(), prefixed, vn, template)
    return _combine(acc1, SitenameCountAccumulator(nested))
end

"""
    map_sitenames_to_rats(sca::SitenameCountAccumulator, varnames_to_ranges::VarNamedTuple)

`sca.entries` contains a mapping of sitenames visited only once, to the corresponding
`VarName`s. Then `varnames_to_ranges` contains a mapping of `VarName`s to the corresponding
`RangeAndTransform` structs.

This function joins the two and returns a `VarNamedTuple` that maps sitenames (visited only
once) to the corresponding `RangeAndTransform` structs.

Resolve each `VarName` tracked by `tracker` against `vnt`, returning a `VarNamedTuple` that
maps sitename-paths to the corresponding values in `vnt`. SeenMultipleTimes sites are left as
[`SeenMultipleTimes`](@ref).
"""
function map_sitenames_to_rats(
    sca::SitenameCountAccumulator,
    varnames_to_rats::VarNamedTuple,
)
    sitenames_to_varnames = sca.entries
    return VarNames.map_values!!(sitenames_to_varnames) do v
        v isa SeenMultipleTimes ? v : varnames_to_rats[v]
    end
end
