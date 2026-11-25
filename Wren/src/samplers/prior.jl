using Random: Random
using FlexiChains: VNChain

using MCMCProgress: Determinate, advance!, chain_at, close_phase!, open_phase!

"""
    prior(
        [rng::Random.AbstractRNG=default_rng()],
        model::Model,
        niters::Integer,
        nchains::Integer=1;
        progress::ProgressSpec=true,
        track=AllQuantities(),
    ) -> FlexiChains.VNChain

Sample from the prior distribution of a Wren model.

Configure `progress` displays by passing a backend, or several as a tuple. The default
`progress=true` uses `TermBackend`, and setting `progress=false` skips progress displays.

By default, all tracked quantities will be included in the returned chain. To include only
one or a selection of tracked quantities, pass a Symbol or a tuple of Symbols; to include
none, pass `track=false` or an empty tuple.
"""
function prior(
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    progress::ProgressSpec=true,
    track=AllQuantities(),
)
    return prior(Random.default_rng(), model, niters, nchains; progress, track)
end
function prior(
    rng::Random.AbstractRNG,
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    progress::ProgressSpec=true,
    track=AllQuantities(),
)
    copy!(Random.TaskLocalRNG(), rng)
    tq_spec = WrenCore.to_tq_spec(track)
    backends = progress_backends(progress)
    show_progress = !isempty(backends)

    t = time()
    label = "Sampling $(model.name) from the prior"
    results = with_progress(backends, label, nchains) do run
        tasks = @sync [
            Threads.@spawn _prior_chain(model, niters, run, j, tq_spec) for j in 1:nchains
        ]
        stack(fetch.(tasks))
    end

    if show_progress
        println("Sampling done in $(round(time() - t, digits=2)) seconds.")
        t = time()
        print("Constructing chain... ")
    end
    chn = WrenCore.to_flexichain(results)
    if show_progress
        println("took $(round(time() - t, digits=2)) seconds")
    end
    return chn
end

function _prior_chain(
    model::Model,
    niters::Integer,
    run,
    j::Integer,
    track::WrenCore.TrackedQuantitySpec,
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
