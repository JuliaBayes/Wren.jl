const VECTOR_VALUE_ACCNAME = :VectorValue

"""
    _get_vector_tval(val, tval, vn, dist, logjac)

Return `tval` with a vector as its stored value.
"""
function _get_vector_tval(
    val,
    tval::TransformedValue{<:AbstractVector,<:AbstractTransform},
    vn,
    dist,
    logjac,
)
    return tval
end

function _get_vector_tval(val, ::TransformedValue{<:Any,AsOriginal}, vn, dist, logjac)
    value = inverse(from_vec(dist))(val)
    return TransformedValue(value, AsVec())
end

# Resolve the overlap between the two methods above for original values which are already
# vectors. The distribution's vectorisation may still be meaningful, so do not assume that
# the raw vector is already in the required representation.
function _get_vector_tval(
    val,
    ::TransformedValue{<:AbstractVector,AsOriginal},
    vn,
    dist,
    logjac,
)
    value = inverse(from_vec(dist))(val)
    return TransformedValue(value, AsVec())
end

function _get_vector_tval(val, tval::TransformedValue, vn, dist, logjac)
    return error(
        "Expected a vectorised or original value for variable $vn, but got a " *
        "TransformedValue with a value of $(tval.value).",
    )
end

"""
    VectorValueAccumulator()

An accumulator that collects parameter values as
`TransformedValue{<:AbstractVector}` objects.

Values already represented as vectors are stored unchanged. Values represented by
[`AsOriginal`](@ref) are vectorised using [`from_vec`](@ref) and stored as [`AsVec`](@ref).
Thus, the stored transform agrees with the selected parameter space while every stored
value has a uniform vector representation.
"""
VectorValueAccumulator() = VNTAccumulator{VECTOR_VALUE_ACCNAME}(_get_vector_tval)

"""
    internal_values_as_vector(values::VarNamedTuple)

Concatenate the internal vectors of the `TransformedValue`s in `values`. An error is
thrown if any stored value is not a vector.
"""
function internal_values_as_vector(values::VarNamedTuple)
    return mapfoldl(pair -> _as_vector(pair.second), vcat, values; init=Union{}[])
end

_as_vector(tval::TransformedValue{<:AbstractVector}) = tval.value
function _as_vector(tval::TransformedValue)
    return error(
        "Expected a TransformedValue containing a vector, but got one containing " *
        "$(tval.value).",
    )
end
