using Random: Random
using ..Wren
using FlexiChains: VNChain

# only for from_samples... could we remove this as a dep in FlexiChains?
using AbstractMCMC: AbstractMCMC

"""
    prior(
        [rng::Random.AbstractRNG=default_rng()],
        model::Wren.Model,
        niters::Integer,
        nchains::Integer=1
    ) -> FlexiChains.VNChain
"""
function prior(model::Wren.Model, niters::Integer, nchains::Integer=1)
    return prior(Random.default_rng(), model, niters, nchains)
end
function prior(rng::Random.AbstractRNG, model::Wren.Model, niters::Integer, nchains::Integer=1)
    # seed task-local from user's rng
    copy!(Random.TaskLocalRNG(), rng)
    tasks = [Threads.@spawn [Draw(FromPrior(Random.TaskLocalRNG()), model) for _ in 1:niters]
             for _ in 1:nchains]
    draws = reduce(hcat, fetch.(tasks))
    return AbstractMCMC.from_samples(VNChain, draws)
end
