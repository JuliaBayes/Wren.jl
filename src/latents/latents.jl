abstract type AbstractLatentSource end

function get_latent end

"""Return a latent source scoped to the variables beneath `vn`."""
function scope end

struct FromPrior{R<:AbstractRNG} <: AbstractLatentSource
    rng::R
end
FromPrior() = FromPrior(default_rng())
get_latent(source::FromPrior, ::VarName, right) = rand(source.rng, right)
scope(source::FromPrior, ::VarName) = source

struct FromParams{P<:VarNamedTuple,S<:Union{AbstractLatentSource,Nothing}} <:
       AbstractLatentSource
    params::P
    fallback::S
end
FromParams(params::VarNamedTuple) = FromParams(params, FromPrior())
function get_latent(source::FromParams, vn::VarName, right)
    if haskey(source.params, vn)
        return source.params[vn]
    end
    source.fallback === nothing && error("No value was provided for variable $vn")
    return get_latent(source.fallback, vn, right)
end

function scope(source::FromParams, vn::VarName)
    child_params = if haskey(source.params, vn)
        params = source.params[vn]
        params isa VarNamedTuple ||
            throw(ArgumentError("Expected the parameters beneath $vn to be a VarNamedTuple"))
        params
    else
        VarNamedTuple()
    end
    child_fallback = source.fallback === nothing ? nothing : scope(source.fallback, vn)
    return FromParams(child_params, child_fallback)
end
