using Random: Random
import DifferentiationInterface as DI
using EnzymeCore: Reverse
using FlexiChains: VNChain
using DynamicHMC: DynamicHMC
using MCMCProgress:
    MCMCProgress, RunProgress, Determinate, advance!, chain_at, close_phase!, open_phase!

mutable struct ChainReporter
    const run::RunProgress
    const chain_index::Int
    phase::Union{MCMCProgress.Phase,Nothing}
end

ChainReporter(run::RunProgress, chain_index::Integer) =
    ChainReporter(run, Int(chain_index), nothing)

function DynamicHMC.make_mcmc_reporter(
    reporter::ChainReporter,
    total_steps::Integer;
    currently_warmup::Bool=false,
    meta...,
)
    close_phases!(reporter)
    reporter.phase = open_phase!(
        chain_at(reporter.run, reporter.chain_index),
        currently_warmup ? "Warmup" : "Sampling",
        Determinate(total_steps),
    )
    return reporter
end

DynamicHMC.report(reporter::ChainReporter, step::Integer; meta...) =
    advance!(reporter.phase, step)

DynamicHMC.report(::ChainReporter, ::AbstractString; meta...) = nothing

function close_phases!(reporter::ChainReporter)
    reporter.phase === nothing || close_phase!(reporter.phase)
    reporter.phase = nothing
    return nothing
end
close_phases!(::DynamicHMC.NoProgressReport) = nothing

"""
    nuts(
        [rng::Random.AbstractRNG=default_rng()],
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Enzyme.Reverse),
        progress::Bool=true,
        reduce_copies::Bool=true,
        gqs::GeneratedQuantitySpec=AllGQs(),
        dynamichmc_kwargs...
    ) -> FlexiChains.VNChain

No U-Turn Sampler (NUTS) for a Wren model. All `dynamichmc_kwargs` are passed to
[`DynamicHMC.mcmc_with_warmup`](@extref), allowing you to customise the warmup and sampling
behaviour. The returned chain holds the generated quantities selected by `gqs`.

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
    gqs::GeneratedQuantitySpec=AllGQs(),
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
        gqs,
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
    gqs::GeneratedQuantitySpec=AllGQs(),
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

    t = time()
    label = "Sampling $(model.name)"
    outputs = with_progress(progress, label, nchains) do run
        tasks = @sync [
            Threads.@spawn _nuts_chain(
                vm_single,
                niters,
                run,
                j;
                gqs,
                dynamichmc_kwargs...,
            ) for j in 1:nchains
        ]
        fetch.(tasks)
    end

    if progress
        println("Sampling done in $(round(time() - t, digits=2)) seconds.")
        t = time()
        print("Constructing chain... ")
    end
    model_results = stack(first.(outputs))
    stats = stack(last.(outputs))
    chn = WrenCore.to_flexichain(model_results, stats)
    if progress
        println("took $(round(time() - t, digits=2)) seconds")
    end
    return chn
end

function _nuts_chain(
    vm_single::WrenCore.VecModel,
    niters::Integer,
    run,
    j::Integer;
    gqs::GeneratedQuantitySpec,
    dynamichmc_kwargs...,
)
    vm = WrenCore._copy_for_thread(vm_single)
    reporter = run === nothing ? DynamicHMC.NoProgressReport() : ChainReporter(run, j)
    results = DynamicHMC.mcmc_with_warmup(
        Random.TaskLocalRNG(),
        vm,
        niters;
        reporter,
        dynamichmc_kwargs...,
    )
    close_phases!(reporter)
    return _nuts_results_and_stats(results, vm, gqs)
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
function _nuts_results_and_stats(
    results::NamedTuple,
    vecmodel::WrenCore.VecModel,
    gqs::GeneratedQuantitySpec,
)
    ϵ = results.ϵ
    model_results =
        [WrenCore.simulate(vecmodel, col; gqs) for col in eachcol(results.posterior_matrix)]
    stats = [_nuts_stats(ts, ϵ) for ts in results.tree_statistics]
    return model_results, stats
end
