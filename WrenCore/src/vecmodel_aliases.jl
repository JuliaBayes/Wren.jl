function _alias_range(mapping, space, group)
    present = filter(vn -> haskey(mapping, vn), group)
    length(present) == 1 || return nothing
    rt = mapping[only(present)]
    rt isa RangeAndTransform && rt.transform isa Union{AsVec,AsUnconstrainedVec} ||
        return nothing
    # Each branch supplies its own distribution transform.
    # Only the vector representation is shared.
    compatible = all(group) do vn
        target = target_transform(space, vn)
        target isa AsOriginal && (target = AsVec())
        isequal(target, rt.transform)
    end
    return compatible ? rt : nothing
end

function _add_aliases(model, mapping, space, aliases)
    groups = aliases === nothing ? _infer_aliases(model) : Tuple(map(Tuple, aliases))
    if aliases === nothing
        # Unsupported leaves and custom transforms retain the ordinary name mapping.
        groups = filter(group -> _alias_range(mapping, space, group) !== nothing, groups)
    end
    used = Set{VarName}()
    for group in groups
        length(group) >= 2 ||
            throw(ArgumentError("An alias group needs at least two names"))
        for vn in group
            vn isa VarName && VarNames.getoptic(vn) isa VarNames.Iden ||
                throw(ArgumentError("Aliases must be plain top-level latent names"))
            vn in used && throw(ArgumentError("Alias groups must not overlap"))
            haskey(model.conditioned, vn) &&
                throw(ArgumentError("Alias members must not be conditioned"))
            push!(used, vn)
        end
        rt = _alias_range(mapping, space, group)
        rt !== nothing || throw(
            ArgumentError(
                "An alias group needs exactly one initial distribution latent and compatible built-in transforms",
            ),
        )
        for vn in group
            mapping = VarNames.setindex!!(mapping, rt, vn)
        end
    end
    return mapping, groups
end

_alias_vector_length(::AsVec, dist) = Plaice.vec_length(dist)
_alias_vector_length(::AsUnconstrainedVec, dist) = Plaice.unconstrained_vec_length(dist)

function _check_alias_block(aliases, vn, dist, rt)
    if any(group -> vn in group, aliases)
        _alias_vector_length(rt.transform, dist) == length(rt.range) ||
            throw(DimensionMismatch("Alias `$vn` does not fit its shared vector slot"))
    end
    return nothing
end

# Immutable counts belong to one evaluation of explicitly declared alias groups.
struct AliasAccumulator{G<:Tuple,N} <: AbstractAccumulator
    aliases::G
    counts::NTuple{N,Int}
end
AliasAccumulator(groups::Tuple) = AliasAccumulator(groups, map(_ -> 0, groups))
accumulator_name(::Type{<:AliasAccumulator}) = :Aliases

function accumulate_assume!!(
    acc::AliasAccumulator,
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    sitename,
)
    counts = map(acc.aliases, acc.counts) do group, count
        count + (vn in group)
    end
    return AliasAccumulator(acc.aliases, counts)
end
accumulate_observe!!(acc::AliasAccumulator, val, vn, right, template, sitename) = acc
reset(acc::AliasAccumulator) = AliasAccumulator(acc.aliases)
split(acc::AliasAccumulator) = reset(acc)
Base.copy(acc::AliasAccumulator) = acc
# Alias declarations apply to this model's plain names, not a child's names.
scope(::AliasAccumulator, ::VarName) = AliasAccumulator(())
combine_with_namespace(parent::AliasAccumulator, ::AliasAccumulator, ::VarName, template) =
    parent
combine(a::AliasAccumulator, b::AliasAccumulator) =
    AliasAccumulator(a.aliases, map(+, a.counts, b.counts))

function _with_alias_check(accs, vi::VecInfo)
    # The stored seed stays at zero because accumulation returns a new immutable value.
    return vi.alias_check === nothing ? accs : setacc!!(accs, vi.alias_check)
end

function _check_aliases(accs::AccumulatorTuple)
    if hasacc(accs, Val(:Aliases))
        all(==(1), getacc(accs, Val(:Aliases)).counts) ||
            throw(ArgumentError("Every alias group must be sampled once per evaluation"))
    end
    return nothing
end
