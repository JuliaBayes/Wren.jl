"""
    FromPrior([rng::AbstractRNG=Random.default_rng()]) <: AbstractLatentSource

Sample latent values from their prior distributions, optionally with a given random number
generator.
"""
struct FromPrior{R<:AbstractRNG} <: AbstractLatentSource
    rng::R
end
FromPrior() = FromPrior(default_rng())
get_latent(source::FromPrior, ::VarName, right, ::Val) =
    TransformedValue(rand(source.rng, right), AsOriginal())
scope(source::FromPrior, ::VarName) = source
