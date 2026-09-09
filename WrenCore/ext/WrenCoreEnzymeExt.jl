module WrenCoreEnzymeExt

using WrenCore
using DifferentiationInterface: DifferentiationInterface as DI
using Enzyme: Enzyme, Reverse, Forward, set_runtime_activity

@inline Enzyme.EnzymeRules.inactive(::typeof(WrenCore._get_range_and_transform), args...) =
    nothing
@inline Enzyme.EnzymeRules.inactive(::typeof(WrenCore.target_transform), args...) = nothing

"""
    _handle_enzyme(vm::VecModel{M,<:AutoEnzyme})

Handle Enzyme-specific quirks so that end users don't have to deal with it.

Specifically, this:

- Runs the gradient once to force compilation of the rules, so that subsequent
  calls are fast. Note that `DI.prepare_gradient` does not actually force rule
  compilation.

- If Enzyme requires runtime activity, and `WrenCore.AUTO_RUNTIME_ACTIVITY[]` is true, then it
  will enable runtime activity (with a warning).
"""
function WrenCore._handle_enzyme(
    vm::WrenCore.VecModel{M,<:DI.AutoEnzyme},
) where {M<:WrenCore.Model}
    try
        lp_and_grad(vm, vm.sample)
    catch e
        if e isa Enzyme.Compiler.EnzymeRuntimeActivityError
            if WrenCore.AUTO_RUNTIME_ACTIVITY[] === false
                rethrow(e)
            else
                if WrenCore.AUTO_RUNTIME_ACTIVITY[] === nothing
                    @info "enabling Enzyme runtime activity for model $(vm.model.name); this may lead to slightly worse performance but should generally not be a cause for concern."
                end
                return WrenCore.VecModel(
                    vm.model,
                    enable_runtime_activity(vm.adtype),
                    vm.lp_function,
                    vm.parameter_space,
                    vm.vecinfo,
                    vm.sample,
                )
            end
        else
            rethrow(e)
        end
    end
    # Successful
    return vm
end

function enable_runtime_activity(::DI.AutoEnzyme{Nothing,FA}) where {FA}
    mode = set_runtime_activity(Reverse)  # promote to reverse mode if unknown.
    return DI.AutoEnzyme{typeof(mode),FA}(mode)
end
function enable_runtime_activity(adt::DI.AutoEnzyme{Mode,FA}) where {Mode,FA}
    mode = set_runtime_activity(adt.mode)
    return DI.AutoEnzyme{typeof(mode),FA}(mode)
end

end # module
