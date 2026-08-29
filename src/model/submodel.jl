"""
    is_submodel(::Type{T}) where {T}

Return true if objects of `T` should be interpreted as submodels when used on the right-hand
side of tilde expressions.

For a custom object `t` to be used as a submodel, `is_submodel(typeof(t))` must return true,
and [`evaluate_submodel`](@ref) must be overloaded for that type.`
"""
is_submodel(::Type) = false
is_submodel(::Type{<:Model}) = true

"""
    evaluate_submodel(
        rhs,
        latent_source::AbstractLatentSource,
        parameter_space::AbstractParameterSpace,
        accs::AccumulatorTuple,
        vn::VarName
    )

Evaluate a tilde-statement of the form `vn ~ rhs`.
"""
function evaluate_submodel(
    rhs,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple,
    # TODO(penelopeysm): allow for no-prefix
    # https://github.com/JuliaBayes/Wren.jl/issues/4
    vn::VarName,
)
    child_source = scope(latent_source, vn)
    child_space = scope(parameter_space, vn)
    child_accs = scope(map(split, accs), vn)
    had_params = haskey(child_accs, Val(PARAMS_ACCNAME))
    # TODO(penelopeysm): Do not unconditionally add ParamsAccumulator here.
    # https://github.com/JuliaBayes/Wren.jl/issues/5
    if !had_params
        child_accs = setacc!!(child_accs, ParamsAccumulator())
    end
    retval, child_accs = evaluate(rhs, child_source, child_space, child_accs)
    child_params = VarNames.parameters(child_accs)
    if !had_params
        child_accs = deleteacc!!(child_accs, Val(PARAMS_ACCNAME))
    end
    return retval, child_params, child_accs
end
