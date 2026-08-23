const SITENAME_TRACKER_ACCNAME = :SiteNameTracker

"""
    Blacklisted

Sentinel stored in a [`SiteNameTracker`](@ref)'s entries in place of a `VarName`, for a
sitename that was hit more than once during model evaluation and thus has no fast path.
"""
struct Blacklisted end

"""
    SiteNameTracker

Accumulator that identifies sampling sites hit exactly once during model evaluation. On
the first hit of a `sitename`, its `VarName` is stored in a `VarNamedTuple` under the key
`VarName{sitename}()`. On any subsequent hit, the entry is replaced with [`Blacklisted`](@ref)
so it is never re-added.

Entries coming from a submodel are nested underneath the submodel's prefix `VarName`, in
exactly the same way as the values of a `VNTAccumulator`. Thus `a ~ inner()`, where `inner`
contains `x ~ Normal()`, leads to an entry at `@vn(a.x)`. This is what lets
[`scope`](@ref) pick out the sites belonging to a submodel without sitenames having to
escape Wren's internals.

After evaluation, `entries` maps each single-hit site's sitename-path to its `VarName`. This
is used to build a fast lookup for ranges and transforms, bypassing the slower
VarNamedTuple path.
"""
struct SiteNameTracker{V<:VarNamedTuple} <: AbstractAccumulator
    entries::V
end

SiteNameTracker() = SiteNameTracker(VarNamedTuple())

accumulator_name(::Type{<:SiteNameTracker}) = SITENAME_TRACKER_ACCNAME

function accumulate_assume!!(
    acc::SiteNameTracker,
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    sitename,
)
    key = VarName{sitename}()
    # A repeat hit blacklists the site. Blacklisting an already-blacklisted site is a no-op,
    # so there is no need to check for that separately.
    new_entry = haskey(acc.entries, key) ? Blacklisted() : vn
    # `key` only ever consists of a single Property optic, so no template is needed.
    new_entries =
        VarNames.templated_setindex!!(acc.entries, new_entry, key, VarNames.NoTemplate())
    return SiteNameTracker(new_entries)
end

accumulate_observe!!(acc::SiteNameTracker, val, vn, right, template, sitename) = acc

reset(acc::SiteNameTracker) = SiteNameTracker()
split(acc::SiteNameTracker) = SiteNameTracker()
Base.copy(acc::SiteNameTracker) = acc

function combine(acc1::SiteNameTracker, acc2::SiteNameTracker)
    # Resolve collisions first, then merge. We can't merge straight away, because `merge`
    # settles a duplicate leaf in favour of its second argument, which would silently pick
    # one of two colliding sites instead of blacklisting them both. But we do have to use
    # `merge` for the combining itself: rebuilding leaf-by-leaf would drop the empty
    # sub-VarNamedTuples that mark submodels which sampled nothing, and `scope` needs those.
    deduped = VarNames.map_pairs!!(acc2.entries) do (key, value)
        haskey(acc1.entries, key) ? Blacklisted() : value
    end
    return SiteNameTracker(merge(acc1.entries, deduped))
end

function combine_with_namespace(
    acc1::SiteNameTracker,
    acc2::SiteNameTracker,
    vn::VarName,
    template,
)
    # Note the absence of an `isempty(acc2.entries)` early-return here: `scope` indexes
    # both VNTs with `vn` unconditionally, so a submodel that sampled nothing still needs an
    # (empty) entry under its prefix. This mirrors `KeepEmptyNamespaces` on VNTAccumulator.
    #
    # Both halves of each entry have to be prefixed: the key, because the sitename now
    # belongs to a submodel, and the value, because it is a submodel-local VarName which
    # has to be resolved against the parent's VarNamedTuple.
    prefixed = VarNames.map_values!!(acc2.entries) do v
        v isa Blacklisted ? v : VarNames.prefix(v, vn)
    end
    nested = VarNames.templated_setindex!!(VarNamedTuple(), prefixed, vn, template)
    return combine(acc1, SiteNameTracker(nested))
end

"""
    sitename_lookup(tracker::SiteNameTracker, vnt::VarNamedTuple)

Resolve each `VarName` tracked by `tracker` against `vnt`, returning a `VarNamedTuple` that
maps sitename-paths to the corresponding values in `vnt`. Blacklisted sites are left as
[`Blacklisted`](@ref).
"""
function sitename_lookup(tracker::SiteNameTracker, vnt::VarNamedTuple)
    return VarNames.map_values!!(tracker.entries) do v
        v isa Blacklisted ? v : vnt[v]
    end
end
