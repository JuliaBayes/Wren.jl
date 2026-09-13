using Random: Random
using FlexiChains: VNChain

# only for from_samples... could we remove this as a dep in FlexiChains?
using AbstractMCMC: AbstractMCMC
using MCMCProgress: Determinate, advance!, chain, closephase!, openphase!

"""
    prior(
        [rng::Random.AbstractRNG=default_rng()],
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        progress::Bool=true,
    ) -> FlexiChains.VNChain

Sample from the prior distribution of a Wren model.
"""
function prior(
    model::WrenCore.Model,
    niters::Integer,
    nchains::Integer=1;
    progress::Bool=true,
)
    return prior(Random.default_rng(), model, niters, nchains; progress)
end
function prior(
    rng::Random.AbstractRNG,
    model::WrenCore.Model,
    niters::Integer,
    nchains::Integer=1;
    progress::Bool=true,
)
    copy!(Random.TaskLocalRNG(), rng)

    t = time()
    label = "Sampling $(model.name) from the prior"
    draws = withprogress(progress, label, nchains) do run
        tasks = [Threads.@spawn _prior_chain(model, niters, run, j) for j in 1:nchains]
        reduce(hcat, fetch.(tasks))
    end

    if progress
        println("Sampling done in $(round(time() - t, digits=2)) seconds.")
        t = time()
        print("Constructing chain... ")
    end
    chn = AbstractMCMC.from_samples(VNChain, draws)
    if progress
        println("took $(round(time() - t, digits=2)) seconds")
    end
    return chn
end

# Draw one chain, reporting it as chain `j` of `run`. A `nothing` run means no
# display was asked for, so nothing is reported.
function _prior_chain(model::WrenCore.Model, niters::Integer, run, j::Integer)
    phase = if run === nothing
        nothing
    else
        openphase!(chain(run, j), "Sampling", Determinate(niters))
    end
    chain_draws = Vector{Draw}(undef, niters)
    for i in 1:niters
        chain_draws[i] = Draw(FromPrior(Random.TaskLocalRNG()), model)
        phase === nothing || advance!(phase, i)
    end
    phase === nothing || closephase!(phase)
    return chain_draws
end
