const VEC_SAMPLE_ACCNAME = :VecSample

"""
    VecSampleAccumulator

An accumulator that holds a full `AbstractVector`, the eventual result of `vector_parameters(vecmodel)` or
similar. This is filled in chunk-by-chunk during model evaluation.
"""
struct VecSampleAccumulator{T<:AbstractVector,R<:VecInfo,Strict} <: AbstractAccumulator
    vals::T
    vecinfo::R
end

VecSampleAccumulator(vals, vecinfo) = VecSampleAccumulator(vals, vecinfo, Val(false))
function VecSampleAccumulator(vals::T, vecinfo::R, ::Val{Strict}) where {T,R,Strict}
    return VecSampleAccumulator{T,R,Strict}(vals, vecinfo)
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
    acc::VecSampleAccumulator{T,R,Strict},
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    sitename::Val,
) where {T,R,Strict}
    rat = _get_range_and_transform(acc.vecinfo, vn, sitename)
    vectorised_tval = _get_vector_tval(val, tval, vn, right, logjac)
    vec_val = get_internal_value(vectorised_tval)
    if Strict
        length(vec_val) == length(rat.range) ||
            throw(DimensionMismatch("Latent `$vn` changed size during vectorisation"))
        eltype(vec_val) === eltype(acc.vals) ||
            throw(ArgumentError("Latent `$vn` requires a common floating-point type"))
    end
    copyto!(acc.vals, first(rat.range), vec_val, 1, length(vec_val))
    return acc
end

function scope(acc::VecSampleAccumulator{T,R,Strict}, vn::VarName) where {T,R,Strict}
    vi = acc.vecinfo
    # TODO(penelopeysm): Reusing acc.vals yields the correct behaviour, but is quite a hack
    # because it relies on the submodel evaluation mutating the parent accumulator. See
    # https://github.com/JuliaBayes/Wren.jl/issues/57.
    return VecSampleAccumulator(
        acc.vals,
        VecInfo(vi.sitename_to_rat[vn], vi.varname_to_rat[vn]),
        Val(Strict),
    )
end

reset(acc::VecSampleAccumulator) = acc
Base.copy(acc::VecSampleAccumulator{T,R,Strict}) where {T,R,Strict} =
    VecSampleAccumulator(copy(acc.vals), acc.vecinfo, Val(Strict))
function combine_from_scope(
    parent::VecSampleAccumulator,
    ::VecSampleAccumulator, # child
    ::VarName,
    template,
)
    # Because the submodel evaluation mutates `parent`, we don't need to use any of the
    # results in `child`.
    return parent
end
