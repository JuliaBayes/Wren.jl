const VEC_SAMPLE_ACCNAME = :VecSample

struct VecSampleAccumulator{T<:AbstractVector,R<:VecInfo} <: AbstractAccumulator
    vals::T
    vecinfo::R
end

accumulator_name(::Type{<:VecSampleAccumulator}) = VEC_SAMPLE_ACCNAME

function accumulate_observe!!(acc::VecSampleAccumulator, val, vn, right, template, sitename)
    return acc
end

function accumulate_assume!!(
    acc::VecSampleAccumulator, val, tval, vn, right, logjac, template, sitename,
)
    rat = _get_range_and_transform(acc.vecinfo, vn, Val(sitename))
    vectorised_tval = _get_vector_tval(val, tval, vn, right, logjac)
    vec_val = get_internal_value(vectorised_tval)
    copyto!(acc.vals, first(rat.range), vec_val, 1, length(vec_val))
    return acc
end

function scope(acc::VecSampleAccumulator, vn::VarName)
    vi = acc.vecinfo
    return VecSampleAccumulator(acc.vals, VecInfo(vi.sitename_to_rat[vn], vi.varname_to_rat[vn]))
end

reset(acc::VecSampleAccumulator) = acc
split(acc::VecSampleAccumulator) = acc
Base.copy(acc::VecSampleAccumulator) = VecSampleAccumulator(copy(acc.vals), acc.vecinfo)
combine(acc1::VecSampleAccumulator, acc2::VecSampleAccumulator) = acc1
function combine_with_namespace(
    parent::VecSampleAccumulator, child::VecSampleAccumulator, ::VarName, template,
)
    return parent
end
