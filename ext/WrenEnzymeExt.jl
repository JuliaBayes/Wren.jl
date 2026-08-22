module WrenEnzymeExt

using Wren
using DifferentiationInterface: DifferentiationInterface as DI
using Enzyme: Enzyme, Reverse, Forward, set_runtime_activity

@inline Enzyme.EnzymeRules.inactive(::typeof(Wren._get_range_and_transform), args...) =
    nothing
@inline Enzyme.EnzymeRules.inactive(::typeof(Wren.target_transform), args...) = nothing

"""
    _handle_enzyme(vm::VecModel{M,<:AutoEnzyme})

Handle Enzyme-specific quirks so that end users don't have to deal with it.

Specifically, this:

- Runs the gradient once to force compilation of the rules, so that subsequent
  calls are fast. Note that `DI.prepare_gradient` does not actually force rule
  compilation.

- If Enzyme requires runtime activity, and `Wren.AUTO_RUNTIME_ACTIVITY[]` is true, then it
  will enable runtime activity (with a warning).
"""
function Wren._handle_enzyme(vm::Wren.VecModel{M,<:DI.AutoEnzyme}) where {M<:Wren.Model}
    try
        lp_and_grad(vm, vm.sample)
    catch e
        if e isa Enzyme.Compiler.EnzymeRuntimeActivityError
            if Wren.AUTO_RUNTIME_ACTIVITY[]
                @warn "enabling runtime activity for Enzyme; this may lead to slightly worse performance."
                return Wren.VecModel(
                    vm.model,
                    enable_runtime_activity(vm.adtype),
                    vm.lp_function,
                    vm.parameter_space,
                    vm.ranges_and_transforms,
                    vm.sample,
                )
            else
                rethrow(e)
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
