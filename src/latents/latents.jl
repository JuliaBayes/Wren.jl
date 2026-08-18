abstract type AbstractLatentSource end

"""
    get_latent(source, vn, dist) -> (value, logjac)

Get latent variable `vn` from `source` and return its value in the distribution's original
space together with the forward log absolute determinant of the Jacobian for the transform
from original space to the source's parameter space.
"""
function get_latent end

"""Return a latent source scoped to the variables beneath `vn`."""
function scope end

struct FromPrior{R<:AbstractRNG} <: AbstractLatentSource
    rng::R
end
FromPrior() = FromPrior(default_rng())
get_latent(source::FromPrior, ::VarName, right) = rand(source.rng, right), zero(LogProbType)
scope(source::FromPrior, ::VarName) = source

struct FromParams{P<:VarNamedTuple,S<:Union{AbstractLatentSource,Nothing}} <:
       AbstractLatentSource
    params::P
    fallback::S
end
FromParams(params::VarNamedTuple) = FromParams(params, nothing)
function get_latent(source::FromParams, vn::VarName, right)
    return if haskey(source.params, vn)
        source.params[vn], zero(LogProbType)
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

abstract type AbstractTransform end
struct AsLinkedVec <: AbstractTransform end
function from_linked_vec end
struct AsVec <: AbstractTransform end
function from_vec end

struct RangeAndTransform{T<:AbstractTransform}
    range::Base.UnitRange
    transform::T
end
struct FromVector{V<:AbstractVector,R<:VarNamedTuple} <: AbstractLatentSource
    params::V
    ranges_and_transforms::R
end
function get_latent(source::FromVector, vn::VarName, right)
    rt = source.ranges_and_transforms[vn]
    range, tfm = rt.range, rt.transform
    vecval = view(source.params, range)
    transform_func = if tfm isa AsLinkedVec
        from_linked_vec(right)
    elseif tfm isa AsVec
        from_vec(right)
    else
        error("todo")
    end
    val, inv_logjac = ChangesOfVariables.with_logabsdet_jacobian(transform_func, vecval)
    return val, -inv_logjac
end
function scope(source::FromVector, vn::VarName)
    return FromVector(source.params, source.ranges_and_transforms[vn])
end
