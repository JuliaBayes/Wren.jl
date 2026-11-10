using Random: Random
using FlexiChains: VNChain

using Term.Progress: ProgressBar, addjob!, update!, start!, stop!, render

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
    if progress
        t = time()
    end

    copy!(Random.TaskLocalRNG(), rng)

    if progress
        pbar = ProgressBar(; columns=:default, transient=false)
        jobs = [addjob!(pbar; N=niters, description="Chain $j/$nchains") for j in 1:nchains]
        progress_channel = Channel{Tuple{Int,Int}}(nchains * 2)
        update_every = max(1, niters ÷ 100)
    end

    tasks = [
        Threads.@spawn begin
            chain_model = WrenCore._copy_for_thread(model)
            chain_results = map(1:niters) do i
                result = WrenCore.simulate(chain_model, Random.TaskLocalRNG())
                if progress && i % update_every == 0
                    put!(progress_channel, (j, i))
                end
                result
            end
            if progress && niters % update_every != 0
                put!(progress_channel, (j, niters))
            end
            chain_results
        end for j in 1:nchains
    ]

    if progress
        start!(pbar)
        chains_done = 0
        while chains_done < nchains
            (j, i) = take!(progress_channel)
            update!(jobs[j]; i=i)
            render(pbar)
            if i == niters
                chains_done += 1
            end
        end
        stop!(pbar)
    end

    if progress
        println("Sampling done in $(round(time() - t, digits=2)) seconds.")
        t = time()
        print("Constructing chain... ")
    end
    chn = WrenCore.to_flexichain(stack(fetch.(tasks)))
    if progress
        println("took $(round(time() - t, digits=2)) seconds")
    end
    return chn
end
