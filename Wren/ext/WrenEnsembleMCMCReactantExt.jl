module WrenEnsembleMCMCReactantExt

using Wren: Wren, WrenCore
using DifferentiationInterface.ADTypes: AutoReactant
import EnsembleMCMC
import Reactant

function _copy_values!(values, output)
    values .= vec(output)
    return nothing
end

Wren._ensemble_vm(model, ::AutoReactant; kwargs...) =
    WrenCore.VecModel(model, nothing; kwargs...)

function Wren._sample_ensemble(rng, vm, initial, niters, ::AutoReactant; move, kwargs...)
    positions = Reactant.to_rarray(reduce(hcat, initial))
    moves = move isa EnsembleMCMC.MoveMixture ? move.moves : (move,)
    width = maximum(
        cld(length(initial), EnsembleMCMC.group_count(moves[i])) for
        i in eachindex(moves) if !(move isa EnsembleMCMC.MoveMixture) || move.weights[i] > 0
    )
    input = Reactant.to_rarray(reduce(hcat, initial[1:width]))
    batch = WrenCore.prepare_batch(vm, input, vm.lp_function)
    scalar = x -> WrenCore.lp(vm, x)
    if Reactant.XLA.platform_name(Reactant.XLA.client(positions)) == "cpu"
        host_batch!(values, points) = copyto!(values, vec(Array(batch(points))))
        target = EnsembleMCMC.BatchedLogDensity(scalar, host_batch!; batch_size=width)
        return EnsembleMCMC.sample(rng, target, initial, niters; move, kwargs...)
    end
    output = batch(input)
    values = similar(positions, eltype(output), width)
    copy_values! = Reactant.Compiler.compile(_copy_values!, (values, output))
    batch!(values, positions) = copy_values!(values, batch(positions))
    target = EnsembleMCMC.BatchedLogDensity(scalar, batch!; batch_size=width)
    history = EnsembleMCMC.sample(
        rng,
        target,
        positions,
        niters;
        move,
        executor=EnsembleMCMC.KernelExecutor(),
        kwargs...,
    )
    return merge(
        history,
        (;
            positions=Array(history.positions),
            logdensities=Array(history.logdensities),
            accepted=Array(history.accepted),
        ),
    )
end

end
