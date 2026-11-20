module WrenEnsembleMCMCExt

using Wren: Wren, WrenCore, Model, FromPrior, FromValues, VarNamedTuple, Draw
using PureRNGs: Philox4x64
using Random: AbstractRNG, default_rng, rand
using FlexiChains: VNChain
import AbstractMCMC
import EnsembleMCMC

function Wren.ensemble(model::Model, niters::Integer, nchains::Integer=1; kwargs...)
    return Wren.ensemble(default_rng(), model, niters, nchains; kwargs...)
end

function Wren.ensemble(
    rng::AbstractRNG,
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    nwalkers::Union{Nothing,Integer}=nothing,
    move=EnsembleMCMC.StretchMove(),
    warmup::Integer=0,
    thin::Integer=1,
    adtype=nothing,
    initial_params::Union{Nothing,AbstractVecOrMat{<:VarNamedTuple}}=nothing,
    progress::Bool=true,
)
    niters > 0 || throw(ArgumentError("niters must be positive"))
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    if initial_params isa AbstractVector
        initial_params = reshape(initial_params, :, 1)
    end
    if !isnothing(initial_params)
        size(initial_params, 1) > 0 && size(initial_params, 2) == nchains ||
            throw(DimensionMismatch("initial_params must have axes (walker, ensemble)"))
    end
    source = isnothing(initial_params) ? FromPrior(rng) : FromValues(initial_params[1, 1])
    template =
        Wren._ensemble_vm(WrenCore._copy_for_thread(model), adtype; latent_source=source)
    count = something(
        nwalkers,
        isnothing(initial_params) ?
        max(4, EnsembleMCMC.minimum_walkers(move, template.dimension)) :
        size(initial_params, 1),
    )
    count > 0 || throw(ArgumentError("nwalkers must be positive"))
    isnothing(initial_params) ||
        size(initial_params, 1) == count ||
        throw(DimensionMismatch("initial_params must have nwalkers rows"))

    draws = map(1:nchains) do chain
        vm = WrenCore._copy_for_thread(template)
        initial = map(1:count) do walker
            if chain == 1 && walker == 1
                copy(template.sample)
            elseif isnothing(initial_params)
                rand(rng, vm)
            else
                WrenCore.vectorise_latents(FromValues(initial_params[walker, chain]), vm)
            end
        end
        key = Philox4x64((rand(rng, UInt64), rand(rng, UInt64)))
        history =
            Wren._sample_ensemble(key, vm, initial, niters, adtype; move, warmup, thin)
        result = [
            Draw(
                view(history.positions, :, walker, i),
                vm,
                (;
                    log_density=history.logdensities[walker, i],
                    accepted=history.accepted[walker, i],
                    move_index=history.move_indices[i],
                    walker_id=history.walker_ids[walker],
                    sweep=warmup + i * thin,
                ),
            ) for i in 1:niters, walker in 1:count
        ]
        progress && println("Completed ensemble $chain/$nchains")
        result
    end
    return [
        AbstractMCMC.from_samples(
            VNChain,
            hcat((view(chain, :, walker) for chain in draws)...),
        ) for walker in 1:count
    ]
end

Wren._ensemble_vm(model, adtype; kwargs...) = WrenCore.VecModel(model, adtype; kwargs...)

function Wren._sample_ensemble(rng, vm, initial, niters, adtype; kwargs...)
    target = x -> WrenCore.lp(vm, x)
    return EnsembleMCMC.sample(rng, target, initial, niters; kwargs...)
end

end
