using Random: Random
import DifferentiationInterface as DI
using EnzymeCore: Reverse
using FlexiChains: VNChain
using DynamicHMC: DynamicHMC
using Term.Progress: ProgressBar, ProgressJob, addjob!, update!, start!, stop!, render

# Channel message types
struct StatusUpdate
    chain::Int
    text::String
end
struct ProgressUpdate
    chain::Int
    step::Int
    total::Int
end
struct ErrorUpdate
    chain::Int
end
const ChannelMsg = Union{StatusUpdate,ProgressUpdate,ErrorUpdate}

# Custom DynamicHMC reporter that sends progress updates through a channel.
mutable struct ChannelReporter
    const channel::Channel{ChannelMsg}
    const chain::Int
    const update_every::Int
end

mutable struct ChannelMCMCReporter
    const channel::Channel{ChannelMsg}
    const chain::Int
    const update_every::Int
    const total_steps::Int
    const currently_warmup::Bool
end

function DynamicHMC.report(r::ChannelReporter, msg::AbstractString; meta...)
    put!(r.channel, StatusUpdate(r.chain, msg))
end
function DynamicHMC.report(r::ChannelMCMCReporter, msg::AbstractString; meta...)
    put!(r.channel, StatusUpdate(r.chain, msg))
end
function DynamicHMC.make_mcmc_reporter(
    r::ChannelReporter,
    total_steps::Integer;
    currently_warmup::Bool=false,
    meta...,
)
    if currently_warmup
        tuning = get(meta, :tuning, "warmup")
        put!(r.channel, StatusUpdate(r.chain, "Warmup ($tuning, $total_steps steps)"))
    end
    ChannelMCMCReporter(r.channel, r.chain, r.update_every, total_steps, currently_warmup)
end
function DynamicHMC.report(r::ChannelMCMCReporter, step::Integer; meta...)
    r.currently_warmup && return nothing
    if step % r.update_every == 0 || step == r.total_steps
        put!(r.channel, ProgressUpdate(r.chain, step, r.total_steps))
    end
end

"""
    nuts(
        [rng::Random.AbstractRNG=default_rng()],
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Enzyme.Reverse),
        progress::Bool=true,
        reduce_copies::Bool=true,
        dynamichmc_kwargs...
    ) -> FlexiChains.VNChain

No U-Turn Sampler (NUTS) for a Wren model. All `dynamichmc_kwargs` are passed to
[`DynamicHMC.mcmc_with_warmup`](@extref), allowing you to customise the warmup and sampling
behaviour.

`reduce_copies` is passed to [`VecModel`](@ref): when sampling with Enzyme, it skips copies
of model arguments that turn out not to be needed (see [`reduce_copies`](@ref)).
"""
function nuts(
    model::WrenCore.Model,
    niters::Integer,
    nchains::Integer=1;
    adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Reverse),
    progress::Bool=true,
    reduce_copies::Bool=true,
    dynamichmc_kwargs...,
)
    return nuts(
        Random.default_rng(),
        model,
        niters,
        nchains;
        adtype,
        progress,
        reduce_copies,
        dynamichmc_kwargs...,
    )
end
function nuts(
    rng::Random.AbstractRNG,
    model::WrenCore.Model,
    niters::Integer,
    nchains::Integer=1;
    adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Reverse),
    progress::Bool=true,
    reduce_copies::Bool=true,
    dynamichmc_kwargs...,
)
    # AD preparation
    if progress
        t = time()
        print("Doing AD preparation... ")
    end
    vm_single = WrenCore.VecModel(model, adtype; reduce_copies)
    if progress
        println("done in $(round(time() - t, digits=2)) seconds.")
        t = time()
    end

    Random.seed!(Random.TaskLocalRNG(), rand(rng, UInt64))

    if progress
        pbar = ProgressBar(; columns=:default, transient=false)
        channel = Channel{ChannelMsg}(nchains * 2)
        update_every = max(1, niters ÷ 100)
    end

    tasks = [
        Threads.@spawn begin
            vm = WrenCore._copy_for_thread(vm_single)
            reporter = if progress
                ChannelReporter(channel, j, update_every)
            else
                DynamicHMC.NoProgressReport()
            end
            try
                results = DynamicHMC.mcmc_with_warmup(
                    Random.TaskLocalRNG(),
                    vm,
                    niters;
                    reporter,
                    dynamichmc_kwargs...,
                )
                _results_and_stats(results, vm)
            catch e
                if progress
                    put!(channel, ErrorUpdate(j))
                end
                rethrow()
            end
        end for j in 1:nchains
    ]

    if progress
        # One status job per chain (shows warmup text), one sampling job per chain
        status_jobs = Dict{Int,ProgressJob}()
        sampling_jobs = Dict{Int,ProgressJob}()
        chains_done = Set{Int}()

        start!(pbar)
        while length(chains_done) < nchains
            msg = take!(channel)
            if msg isa ErrorUpdate
                push!(chains_done, msg.chain)
            elseif msg isa StatusUpdate
                j = msg.chain
                if !haskey(status_jobs, j)
                    status_jobs[j] =
                        addjob!(pbar; N=1, description="Chain $j/$nchains: $(msg.text)")
                else
                    status_jobs[j].description = "Chain $j/$nchains: $(msg.text)"
                end
            elseif msg isa ProgressUpdate
                j = msg.chain
                if !haskey(sampling_jobs, j)
                    # Warmup is done; mark status job as complete
                    if haskey(status_jobs, j)
                        status_jobs[j].description = "Chain $j/$nchains: Warmup done"
                        update!(status_jobs[j]; i=1)
                    end
                    sampling_jobs[j] = addjob!(
                        pbar;
                        N=msg.total,
                        description="Sampling (chain $j/$nchains)",
                    )
                end
                update!(sampling_jobs[j]; i=msg.step)
                if msg.step == msg.total
                    push!(chains_done, j)
                end
            end
            render(pbar)
        end
        stop!(pbar)
    end

    if progress
        println("Sampling done in $(round(time() - t, digits=2)) seconds.")
        t = time()
        print("Constructing chain... ")
    end
    outputs = fetch.(tasks)
    # `stack` (unlike `reduce(hcat, ...)`) still gives a matrix when there is one chain.
    model_results = stack(first.(outputs))
    stats = stack(last.(outputs))
    chn = WrenCore.to_flexichain(model_results, stats)
    if progress
        println("took $(round(time() - t, digits=2)) seconds")
    end
    return chn
end

function _nuts_stats(tree_stat, step_size)
    t = tree_stat.termination
    return (;
        step_size,
        tree_depth=tree_stat.depth,
        is_divergent=DynamicHMC.is_divergent(t),
        is_max_depth=(t == DynamicHMC.REACHED_MAX_DEPTH),
        acceptance_rate=tree_stat.acceptance_rate,
        log_density=tree_stat.π,
    )
end
function _results_and_stats(results::NamedTuple, vecmodel::WrenCore.VecModel)
    ϵ = results.ϵ
    model_results =
        [WrenCore.simulate(vecmodel, col) for col in eachcol(results.posterior_matrix)]
    stats = [_nuts_stats(ts, ϵ) for ts in results.tree_statistics]
    return model_results, stats
end
