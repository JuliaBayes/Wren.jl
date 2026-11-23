using Random: Random
using FlexiChains: VNChain

using MCMCProgress: Determinate, advance!, chain_at, close_phase!, open_phase!

"""
    prior(
        [rng::Random.AbstractRNG=default_rng()],
        model::Model,
        niters::Integer,
        nchains::Integer=1;
        progress::Bool=true,
        track=true,
    ) -> FlexiChains.VNChain

Sample from the prior distribution of a Wren model. Tracked quantities selected by `track`
(see [`to_tq_spec`](@ref)) will be included in the returned chain.
"""
function prior(
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    progress::Bool=true,
    track=true,
)
    return prior(Random.default_rng(), model, niters, nchains; progress, track)
end
function prior(
    rng::Random.AbstractRNG,
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    progress::Bool=true,
    track=true,
)
    copy!(Random.TaskLocalRNG(), rng)
    tq_spec = to_tq_spec(track)

    t = time()
    label = "Sampling $(model.name) from the prior"
    results = with_progress(progress, label, nchains) do run
        tasks = @sync [
            Threads.@spawn _prior_chain(model, niters, run, j, tq_spec) for j in 1:nchains
        ]
        stack(fetch.(tasks))
    end

    if progress
        println("Sampling done in $(round(time() - t, digits=2)) seconds.")
        t = time()
        print("Constructing chain... ")
    end
    chn = WrenCore.to_flexichain(results)
    if progress
        println("took $(round(time() - t, digits=2)) seconds")
    end
    return chn
end

function _prior_chain(
    model::Model,
    niters::Integer,
    run,
    j::Integer,
    track::TrackedQuantitySpec,
)
    chain_model = WrenCore._copy_for_thread(model)
    phase = if run === nothing
        nothing
    else
        open_phase!(chain_at(run, j), "Sampling", Determinate(niters))
    end
    chain_results = map(1:niters) do i
        result = WrenCore.simulate(chain_model, Random.TaskLocalRNG(); track)
        phase === nothing || advance!(phase, i)
        result
    end
    phase === nothing || close_phase!(phase)
    return chain_results
end
