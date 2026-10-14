"""
    FromValues(
        params::VarNamedTuple,
        fallback::Union{AbstractLatentSource,Nothing}=nothing
    ) <: AbstractLatentSource

Take latent values from the given `VarNamedTuple`. If the `VarNamedTuple` does not contain a
value for a variable, then the `fallback` source is used, or an error thrown if `fallback`
is `nothing`.
"""
struct FromValues{P<:VarNamedTuple,S<:Union{AbstractLatentSource,Nothing}} <:
       AbstractLatentSource
    params::P
    fallback::S
end
FromValues(params::VarNamedTuple) = FromValues(params, nothing)
function get_latent(source::FromValues, vn::VarName, right, sitename::Val)
    return if haskey(source.params, vn)
        value = source.params[vn]
        value isa TransformedValue ? value : TransformedValue(value, AsOriginal())
    else
        source.fallback === nothing && error("No value was provided for variable $vn")
        get_latent(source.fallback, vn, right, sitename)
    end
end
function scope(source::FromValues, vn::VarName)
    child_fallback = if source.fallback === nothing
        nothing
    else
        scope(source.fallback, vn)
    end
    return if haskey(source.params, vn)
        child_params = source.params[vn]
        child_params isa VarNamedTuple ||
            throw(ArgumentError("the parameter $vn must contain a VarNamedTuple"))
        FromValues(child_params, child_fallback)
    else
        something(child_fallback, FromValues(VarNamedTuple()))
    end
end
