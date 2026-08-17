abstract type AbstractLatentSource end

function get_latent end

struct FromPrior{R<:AbstractRNG} <: AbstractLatentSource
    rng::R
end
FromPrior() = FromPrior(default_rng())
get_latent(source::FromPrior, ::VarName, right) = rand(source.rng, right)

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
