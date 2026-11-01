module WrenPathfinderExt

using Wren: Wren, Model, DEFAULT_AD, FromPrior, logdensity, Draw
using Accessors: @set
using AbstractMCMC: AbstractMCMC
using FlexiChains: VNChain
using Random: AbstractRNG
import Pathfinder

function Wren.pathfinder(
    rng::AbstractRNG,
    model::Model,
    ndraws::Integer;
    nruns::Int=4,
    init=nothing,
    adtype=DEFAULT_AD,
    latent_source=FromPrior(rng),
    reduce_copies=true,
    ntasks::Int=1,
    ntasks_per_run::Int=1,
    kwargs...,
)
    ntasks == 1 && ntasks_per_run == 1 ||
        throw(ArgumentError("Wren models require ntasks=1 and ntasks_per_run=1."))
    density = logdensity(
        Wren.WrenCore._copy_for_thread(model),
        adtype;
        latent_source,
        reduce_copies,
    )
    if init !== nothing
        nruns = length(init)
    end
    result = Pathfinder.multipathfinder(
        density,
        Int(ndraws);
        rng,
        nruns,
        init,
        ntasks,
        ntasks_per_run,
        kwargs...,
    )
    draws = map(eachcol(result.draws)) do params
        Draw(params, density)
    end
    chain = AbstractMCMC.from_samples(VNChain, reshape(draws, :, 1))
    return @set result.draws_transformed = chain
end

end
