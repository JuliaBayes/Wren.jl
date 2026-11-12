"""
    FromValues(
        values::VarNamedTuple,
        fallback::AbstractLatentSource=Void()
    ) <: AbstractLatentSource

Take latent values from the given `VarNamedTuple`. If the `VarNamedTuple` does not contain a
value for a variable, then the `fallback` source is used. The default fallback, `Void()`,
throws an error.
"""
struct FromValues{P<:VarNamedTuple,S<:AbstractLatentSource} <: AbstractLatentSource
    values::P
    fallback::S
end
FromValues(values::VarNamedTuple) = FromValues(values, Void())
function get_latent(source::FromValues, vn::VarName, right, sitename::Val)
    return if haskey(source.values, vn)
        value = source.values[vn]
        value isa TransformedValue ? value : TransformedValue(value, AsOriginal())
    else
        get_latent(source.fallback, vn, right, sitename)
    end
end
function scope(source::FromValues, vn::VarName)
    child_fallback = scope(source.fallback, vn)
    return if haskey(source.values, vn)
        child_values = source.values[vn]
        child_values isa VarNamedTuple ||
            throw(ArgumentError("the parameter $vn must contain a VarNamedTuple"))
        FromValues(child_values, child_fallback)
    else
        child_fallback
    end
end
