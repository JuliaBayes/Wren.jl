using Random: Random
import DifferentiationInterface as DI
using EnzymeCore: Reverse
using FlexiChains: VNChain
using DynamicHMC: DynamicHMC
using MCMCProgress:
    MCMCProgress, RunProgress, Determinate, advance!, chain_at, close_phase!, open_phase!

abstract type AbstractInitialization end

"""
    FindFiniteDensity(source; max_attempts::Integer=1)

Initialise each NUTS chain by drawing from `source` until its unconstrained coordinates and
log-density are finite, trying at most `max_attempts` candidates per chain. `source` is
converted with [`WrenCore.to_latent_source`](@ref). A failed search throws before warmup.

The same source supplies [`VecModel`](@ref) preparation values. Preparation draws do not
count towards `max_attempts`; errors during preparation or model evaluation propagate.
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
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Enzyme.Reverse),
        progress::Bool=true,
        reduce_copies::Bool=true,
        track::TrackedQuantitySpec=AllQuantities(),
        init::Union{Nothing,AbstractInitialization}=nothing,
        initialization=nothing,
        dynamichmc_kwargs...
    ) -> FlexiChains.VNChain

No U-Turn Sampler (NUTS) for a Wren model. All `dynamichmc_kwargs` are passed to
[`DynamicHMC.mcmc_with_warmup`](@extref), allowing you to customise the warmup and sampling
behaviour. The returned chain holds the tracked quantities selected by `track`.

Each chain starts from an independent prior draw by default.

Use `init=FindFiniteDensity(FromValues(VarNamedTuple(; x=1.0)))` to supply initial values in
model space. [`FindFiniteDensity`](@ref) checks one candidate per chain by default; increase
`max_attempts` to retry non-finite starting points or log-densities. For opt-in interval
initialisation, use `init=FindFiniteDensity(WrenCore.FromUnconstrainedInterval())` and set
the attempt bound as needed.

DynamicHMC's warmup and step-size options, including `initialization=(; ϵ=0.1)`, remain
available. Passing raw `initialization.q` or `initialization.κ` is deprecated. For direct
control of unconstrained positions or metrics, construct a [`VecModel`](@ref) and pass it
to [`DynamicHMC.mcmc_with_warmup`](@extref). A raw `initialization.q` cannot be combined
with `init`, since both specify the starting point.

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
    track::TrackedQuantitySpec=AllQuantities(),
    init::Union{Nothing,AbstractInitialization}=nothing,
    initialization=nothing,
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
        initialization,
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
    track::TrackedQuantitySpec=AllQuantities(),
    init::Union{Nothing,AbstractInitialization}=nothing,
    initialization=nothing,
    dynamichmc_kwargs...,
)
    initialization = initialization === nothing ? nothing : (; initialization...)
    init !== nothing &&
        initialization !== nothing &&
        haskey(initialization, :q) &&
        throw(
            ArgumentError("`init` and `initialization.q` both specify the starting point."),
        )
    if initialization !== nothing &&
       (haskey(initialization, :q) || haskey(initialization, :κ))
        Base.depwarn(
            "Passing `initialization.q` or `initialization.κ` to `nuts` is deprecated. Use a Wren initialization policy, or construct a VecModel and call DynamicHMC directly for unconstrained positions or metrics.",
            :nuts,
        )
    end
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
    chain_initializations = if init === nothing
        fill(initialization, nchains)
    else
        options = initialization === nothing ? (;) : initialization
        [(; options..., q=_initial_position(vm_single, init)) for _ in 1:nchains]
    end

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
                initialization=chain_initializations[j],
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
    initialization=nothing,
    dynamichmc_kwargs...,
)
    vm = WrenCore._copy_for_thread(vm_single)
    rng = Random.TaskLocalRNG()
    if initialization === nothing
        initialization = (; q=WrenCore.vector_parameters(vm, WrenCore.FromPrior(rng)))
    end
    reporter = run === nothing ? DynamicHMC.NoProgressReport() : ChainReporter(run, j)
    results = DynamicHMC.mcmc_with_warmup(
        rng,
        vm,
        niters;
        reporter,
        initialization,
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
