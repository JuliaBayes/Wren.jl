const SITENAME_TRACKER_ACCNAME = :SiteNameTracker

"""
    SiteNameTracker

Accumulator that identifies sampling sites hit exactly once during model evaluation. On
the first hit of a `sitename`, its `VarName` is stored in a NamedTuple keyed by
`sitename`. On any subsequent hit, the entry is removed and the `sitename` is blacklisted
so it is never re-added.

After evaluation, `entries` contains only the single-hit sites, mapping
`sitename => VarName`. This is used to build a fast NamedTuple lookup for ranges and
transforms, bypassing the slower VarNamedTuple path.
"""
struct SiteNameTracker{NT<:NamedTuple,BL<:Tuple} <: AbstractAccumulator
    entries::NT
    blacklist::BL
end

SiteNameTracker() = SiteNameTracker(NamedTuple(), ())

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
    if sitename in acc.blacklist
        return acc
    elseif haskey(acc.entries, sitename)
        new_entries = Base.structdiff(acc.entries, NamedTuple{(sitename,)})
        return SiteNameTracker(new_entries, (acc.blacklist..., sitename))
    else
        new_entries = merge(acc.entries, NamedTuple{(sitename,)}((vn,)))
        return SiteNameTracker(new_entries, acc.blacklist)
    end
end

accumulate_observe!!(acc::SiteNameTracker, val, vn, right, template, sitename) = acc

reset(acc::SiteNameTracker) = SiteNameTracker()
split(acc::SiteNameTracker) = SiteNameTracker()
Base.copy(acc::SiteNameTracker) = acc

function combine(acc1::SiteNameTracker, acc2::SiteNameTracker)
    all_blacklisted = (acc1.blacklist..., acc2.blacklist...)
    combined = NamedTuple()
    for (k, v) in pairs(merge(acc1.entries, acc2.entries))
        if k in all_blacklisted
            continue
        elseif haskey(acc1.entries, k) && haskey(acc2.entries, k)
            all_blacklisted = (all_blacklisted..., k)
        else
            combined = merge(combined, NamedTuple{(k,)}((v,)))
        end
    end
    return SiteNameTracker(combined, all_blacklisted)
end

function combine_with_namespace(
    acc1::SiteNameTracker,
    acc2::SiteNameTracker,
    ::VarName,
    template,
)
    return combine(acc1, acc2)
end
