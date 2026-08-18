abstract type AbstractLatentSource end

"""
    get_latent(source, vn, dist) -> TransformedValue

Get latent variable `vn` from `source`, retaining information about the representation in
which the source supplied it. The parameter space subsequently converts this to the raw
value used in the model and computes the relevant log-Jacobian.
"""
function get_latent end

"""Return a latent source scoped to the variables beneath `vn`."""
function scope end

struct FromPrior{R<:AbstractRNG} <: AbstractLatentSource
    rng::R
end
FromPrior() = FromPrior(default_rng())
get_latent(source::FromPrior, ::VarName, right) =
    TransformedValue(rand(source.rng, right), AsOriginal())
scope(source::FromPrior, ::VarName) = source

struct FromParams{P<:VarNamedTuple,S<:Union{AbstractLatentSource,Nothing}} <:
       AbstractLatentSource
    params::P
    fallback::S
end
FromParams(params::VarNamedTuple) = FromParams(params, nothing)
function get_latent(source::FromParams, vn::VarName, right)
    return if haskey(source.params, vn)
        value = source.params[vn]
        value isa TransformedValue ? value : TransformedValue(value, AsOriginal())
    else
        source.fallback === nothing && error("No value was provided for variable $vn")
        get_latent(source.fallback, vn, right)
    end
end
function scope(source::FromParams, vn::VarName)
    child_fallback = if source.fallback === nothing
        nothing
    else
        scope(source.fallback, vn)
    end
    return if haskey(source.params, vn)
        child_params = source.params[vn]
        child_params isa VarNamedTuple ||
            throw(ArgumentError("the parameter $vn must contain a VarNamedTuple"))
        FromParams(child_params, child_fallback)
    else
        child_fallback
    end
end

struct RangeAndTransform{R<:Base.UnitRange,T<:AbstractTransform}
    range::R
    transform::T
end
Base.:(==)(a::RangeAndTransform, b::RangeAndTransform) =
    a.range == b.range && a.transform == b.transform
Base.isequal(a::RangeAndTransform, b::RangeAndTransform) =
    isequal(a.range, b.range) && isequal(a.transform, b.transform)

struct FromVector{V<:AbstractVector,R<:VarNamedTuple} <: AbstractLatentSource
    params::V
    ranges_and_transforms::R
end

_get_range_and_transform(ranges_and_transforms::VarNamedTuple, vn::VarName) =
    ranges_and_transforms[vn]

function get_latent(source::FromVector, vn::VarName, right)
    rt = _get_range_and_transform(source.ranges_and_transforms, vn)
    range, tfm = rt.range, rt.transform
    return TransformedValue(view(source.params, range), tfm)
end
function scope(source::FromVector, vn::VarName)
    return FromVector(source.params, source.ranges_and_transforms[vn])
end

scope(parameter_space::Union{UnconstrainedSpace,OriginalSpace}, ::VarName) = parameter_space
