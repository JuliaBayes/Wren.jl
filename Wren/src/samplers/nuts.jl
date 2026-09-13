using Random: Random
import DifferentiationInterface as DI
using EnzymeCore: Reverse
using FlexiChains: VNChain
using AbstractMCMC: AbstractMCMC
using DynamicHMC: DynamicHMC
using MCMCProgress: MCMCProgress, Determinate, advance!, chain, closephase!, openphase!

"""
    ChainReporter(run, index)

A `DynamicHMC.report` sink that turns the warmup and sampling stages of one
chain into phases on chain `index` of the MCMCProgress run `run`.

DynamicHMC opens a stage by asking for a reporter and never announces that one
is over, so each stage's phase is closed when the next stage opens; the last is
closed by `closephases!` once sampling returns.
"""
mutable struct ChainReporter
    const run::MCMCProgress.Run
    const index::Int
    phase::Union{MCMCProgress.Phase,Nothing}
end

ChainReporter(run::MCMCProgress.Run, index::Integer) =
    ChainReporter(run, Int(index), nothing)

function DynamicHMC.make_mcmc_reporter(
    reporter::ChainReporter,
    total_steps::Integer;
    currently_warmup::Bool=false,
    meta...,
)
    # A phase name shares the width of a bar with the bar itself, so it stays
    # short: DynamicHMC's warmup runs as a sequence of windows, and the total
    # each successive "Warmup" phase declares is what tells them apart.
    name = currently_warmup ? "Warmup" : "Sampling"
    closephases!(reporter)
    reporter.phase =
        openphase!(chain(reporter.run, reporter.index), name, Determinate(total_steps))
    return reporter
end

DynamicHMC.report(reporter::ChainReporter, step::Integer; meta...) =
    advance!(reporter.phase, step)

# Stage messages such as "found initial stepsize" carry no position, and the
# stage they belong to already has a phase of its own.
DynamicHMC.report(::ChainReporter, ::AbstractString; meta...) = nothing

"""
    closephases!(reporter)

Close the phase `reporter` currently has open, if any. A run whose phases are
all closed is one whose display shows no chain still working.
"""
function closephases!(reporter::ChainReporter)
    reporter.phase === nothing || closephase!(reporter.phase)
    reporter.phase = nothing
    return nothing
end
closephases!(::DynamicHMC.NoProgressReport) = nothing

"""
    nuts(
        [rng::Random.AbstractRNG=default_rng()],
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Enzyme.Reverse),
        progress::Bool=true,
        dynamichmc_kwargs...
    ) -> FlexiChains.VNChain

No U-Turn Sampler (NUTS) for a Wren model. All `dynamichmc_kwargs` are passed to
[`DynamicHMC.mcmc_with_warmup`](@extref), allowing you to customise the warmup and sampling
behaviour.
"""
function nuts(
    model::WrenCore.Model,
    niters::Integer,
    nchains::Integer=1;
    adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Reverse),
    progress::Bool=true,
    dynamichmc_kwargs...,
)
    return nuts(
        Random.default_rng(),
        model,
        niters,
        nchains;
        adtype,
        progress,
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
    dynamichmc_kwargs...,
)
    # AD preparation
    if progress
        t = time()
        print("Doing AD preparation... ")
    end
    vm_single = WrenCore.VecModel(model, adtype)
    if progress
        println("done in $(round(time() - t, digits=2)) seconds.")
    end

    copy!(Random.TaskLocalRNG(), rng)

    t = time()
    label = "Sampling $(model.name)"
    draws = withprogress(progress, label, nchains) do run
        tasks = [
            Threads.@spawn _nuts_chain(vm_single, niters, run, j; dynamichmc_kwargs...) for j in 1:nchains
        ]
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

# Sample one chain, reporting it as chain `j` of `run`. A `nothing` run means no
# display was asked for, so nothing is reported.
function _nuts_chain(
    vm_single::WrenCore.VecModel,
    niters::Integer,
    run,
    j::Integer;
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
    closephases!(reporter)
    return to_draws(results, vm)
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
function to_draws(results::NamedTuple, vecmodel::WrenCore.VecModel)
    ϵ = results.ϵ
    return [
        Draw(col, vecmodel, _nuts_stats(ts, ϵ)) for
        (col, ts) in zip(eachcol(results.posterior_matrix), results.tree_statistics)
    ]
end
