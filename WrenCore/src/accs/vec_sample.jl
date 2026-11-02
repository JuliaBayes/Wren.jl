const VEC_SAMPLE_ACCNAME = :VecSample

"""
    VecSampleAccumulator

An accumulator that holds a full `AbstractVector`, the eventual result of `rand(ldf)` or
similar. This is filled in chunk-by-chunk during model evaluation.
"""
struct VecSampleAccumulator{T<:AbstractVector,R<:VecInfo} <: AbstractAccumulator
    vals::T
    vecinfo::R
end

accumulator_name(::Type{<:VecSampleAccumulator}) = VEC_SAMPLE_ACCNAME

function accumulate_observe!!(
    acc::VecSampleAccumulator,
    val,
    vn,
    right,
    template,
    @nospecialize(sitename::Val),
)
    return acc
end

function accumulate_assume!!(
    acc::VecSampleAccumulator,
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    sitename::Val,
)
    rat = _get_range_and_transform(acc.vecinfo, vn, sitename)
    _check_alias_block(acc.vecinfo.aliases, vn, right, rat)
    vectorised_tval = _get_vector_tval(val, tval, vn, right, logjac)
    vec_val = get_internal_value(vectorised_tval)
    copyto!(acc.vals, first(rat.range), vec_val, 1, length(vec_val))
    return acc
end

function scope(acc::VecSampleAccumulator, vn::VarName)
    vi = acc.vecinfo
    # TODO(penelopeysm): Reusing acc.vals yields the correct behaviour, but is quite a hack
    # because it relies on the submodel evaluation mutating the parent accumulator. See
    # https://github.com/JuliaBayes/Wren.jl/issues/57.
    return VecSampleAccumulator(
        acc.vals,
        VecInfo(vi.sitename_to_rat[vn], vi.varname_to_rat[vn]),
    )
end

reset(acc::VecSampleAccumulator) = acc
split(acc::VecSampleAccumulator) = acc
Base.copy(acc::VecSampleAccumulator) = VecSampleAccumulator(copy(acc.vals), acc.vecinfo)
combine(acc1::VecSampleAccumulator, ::VecSampleAccumulator) = acc1
function combine_with_namespace(
    parent::VecSampleAccumulator,
    ::VecSampleAccumulator, # child
    ::VarName,
    template,
)
    # Because the submodel evaluation mutates `parent`, we don't need to use any of the
    # results in `child`.
    return parent
end
