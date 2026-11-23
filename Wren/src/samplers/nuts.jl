using Random: Random
import DifferentiationInterface as DI
using EnzymeCore: Reverse
using FlexiChains: VNChain
using DynamicHMC: DynamicHMC
using MCMCProgress:
    MCMCProgress, RunProgress, Determinate, advance!, chain_at, close_phase!, open_phase!

"""
    AbstractInitialization

An abstract type for ways to initialise MCMC samplers.

Currently, only [`FindFiniteDensity`](@ref) is provided.
"""
abstract type AbstractInitialization end

"""
    FindFiniteDensity(source; max_attempts::Integer=1)

Initialise a chain by drawing from `source` until its unconstrained coordinates and
log-density are finite, trying at most `max_attempts` candidates per chain.

`source` can either be a [`WrenCore.AbstractLatentSource`](@ref), or something that can
be converted to it. In particular:

- `source::Random.AbstractRNG` is shorthand for `FromPrior(source)`, i.e., draw from the
  prior with the given RNG.
- `source::VarNamedTuple` is shorthand for `FromValues(source)`, i.e., use the given values
  *(in model space)* as the initial point.

To use Stan-style initialisation from a uniform interval, use
[`source=FromUnconstrainedInterval([rng, ]lower, upper)`](@ref FromUnconstrainedInterval).

If initialisation fails after `max_attempts`, an error is thrown.
"""
struct FindFiniteDensity{S<:WrenCore.AbstractLatentSource} <: AbstractInitialization
    source::S
    max_attempts::Int

    function FindFiniteDensity(
        source::S,
        max_attempts::Integer,
    ) where {S<:WrenCore.AbstractLatentSource}
        max_attempts > 0 || throw(ArgumentError("`max_attempts` must be positive."))
        return new{S}(source, Int(max_attempts))
    end
end

FindFiniteDensity(source; max_attempts::Integer=1) =
    FindFiniteDensity(WrenCore.to_latent_source(source), max_attempts)

_initial_source(init::FindFiniteDensity) = init.source

function _initial_position(vm::WrenCore.VecModel, init::FindFiniteDensity)
    for _ in 1:init.max_attempts
        q = WrenCore.vector_parameters(vm, init.source)
        all(isfinite, q) && isfinite(WrenCore.lp(vm, q)) && return q
    end
    throw(
        ArgumentError(
            "No finite-density initial point found after $(init.max_attempts) attempts.",
        ),
    )
end

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
        model::Model,
        niters::Integer,
        nchains::Integer=1;
        adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Enzyme.Reverse),
        progress::Bool=true,
        reduce_copies::Bool=true,
        track::TrackedQuantitySpec=AllQuantities(),
        init::Union{Nothing,AbstractInitialization}=nothing,
        initial_stepsize::Union{Nothing,Real}=nothing,
        dynamichmc_kwargs...
    ) -> FlexiChains.VNChain

No U-Turn Sampler (NUTS) for a Wren model. All `dynamichmc_kwargs` are passed to
[`DynamicHMC.mcmc_with_warmup`](@extref), allowing you to customise the warmup and sampling
behaviour.

Each chain starts from an independent prior draw by default. Use `init` to select an
initialization policy. `initial_stepsize` replaces the default initial step-size search.

`reduce_copies` is passed to [`VecModel`](@ref): when sampling with Enzyme, it skips copies
of model arguments that turn out not to be needed (see [`reduce_copies`](@ref)).

Tracked quantities selected by `track` will be included in the returned chain. Note that
tracked quantities are not evaluated during the log-density step, but only once per iteration.
See also: [`AllQuantities`](@ref), [`SomeQuantities`](@ref), [`NoQuantities`](@ref)
"""
function nuts(
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Reverse),
    progress::Bool=true,
    reduce_copies::Bool=true,
    track::TrackedQuantitySpec=AllQuantities(),
    init::Union{Nothing,AbstractInitialization}=nothing,
    initial_stepsize::Union{Nothing,Real}=nothing,
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
        track,
        init,
        initial_stepsize,
        dynamichmc_kwargs...,
    )
end
function nuts(
    rng::Random.AbstractRNG,
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Reverse),
    progress::Bool=true,
    reduce_copies::Bool=true,
    track::TrackedQuantitySpec=AllQuantities(),
    init::Union{Nothing,AbstractInitialization}=nothing,
    initial_stepsize::Union{Nothing,Real}=nothing,
    dynamichmc_kwargs...,
)
    haskey(dynamichmc_kwargs, :initialization) && throw(
        ArgumentError(
            "Use `init` for starting points and `initial_stepsize` for the step size.",
        ),
    )
    initial_stepsize = initial_stepsize === nothing ? nothing : Float64(initial_stepsize)
    source = init === nothing ? WrenCore.FromPrior() : _initial_source(init)
    # AD preparation
    if progress
        t = time()
        print("Doing AD preparation... ")
    end
    vm_single = WrenCore.VecModel(model, adtype; reduce_copies, latent_source=source)
    if progress
        println("done in $(round(time() - t, digits=2)) seconds.")
        t = time()
    end

    Random.seed!(Random.TaskLocalRNG(), rand(rng, UInt64))

    # Sources can contain mutable RNGs, so draw initial points before spawning tasks.
    starts =
        init === nothing ? nothing : [_initial_position(vm_single, init) for _ in 1:nchains]

    t = time()
    label = "Sampling $(model.name)"
    outputs = with_progress(progress, label, nchains) do run
        tasks = @sync [
            Threads.@spawn _nuts_chain(
                vm_single,
                niters,
                run,
                j;
                track,
                q=starts === nothing ? nothing : starts[j],
                initial_stepsize,
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
    track::TrackedQuantitySpec,
    q,
    initial_stepsize,
    warmup_stages=initial_stepsize === nothing ? DynamicHMC.default_warmup_stages() :
                  DynamicHMC.default_warmup_stages(; stepsize_search=nothing),
    dynamichmc_kwargs...,
)
    vm = WrenCore._copy_for_thread(vm_single)
    rng = Random.TaskLocalRNG()
    if q === nothing
        q = WrenCore.vector_parameters(vm, WrenCore.FromPrior(rng))
    end
    reporter = run === nothing ? DynamicHMC.NoProgressReport() : ChainReporter(run, j)
    results = DynamicHMC.mcmc_with_warmup(
        rng,
        vm,
        niters;
        reporter,
        initialization=(; q, ϵ=initial_stepsize),
        warmup_stages,
        dynamichmc_kwargs...,
    )
    close_phases!(reporter)
    return _nuts_results_and_stats(results, vm, track)
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
    track::TrackedQuantitySpec,
)
    ϵ = results.ϵ
    model_results = [
        WrenCore.simulate(vecmodel, col; track) for col in eachcol(results.posterior_matrix)
    ]
    stats = [_nuts_stats(ts, ϵ) for ts in results.tree_statistics]
    return model_results, stats
end
