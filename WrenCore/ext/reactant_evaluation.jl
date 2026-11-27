# Branch tracing needs tensor slices, not SubArrays with untraceable offset fields.
function WrenCore.get_latent(
    source::WrenCore.FromVector{<:Reactant.TracedRArray},
    vn::WrenCore.VarName,
    right,
    sitename::Val,
)
    rt = WrenCore._get_range_and_transform(source.vecinfo, vn, sitename)
    return WrenCore.TransformedValue(source.params[rt.range], rt.transform)
end

# Base.view does not accept traced scalar indices. Tensor indexing handles both ranks.
# https://github.com/EnzymeAD/Reactant.jl/issues/3830
WrenCore._maybe_view(value::Reactant.TracedRArray, indices...) = value[indices...]

function _reactant_evaluation_fn(model::WrenCore.Model)
    fn_args = WrenCore._evaluation_fn_args()
    expr = WrenCore.evaluation_fn_expr(keys(model.args), model.stmts; fn_args)
    expr.args[2] = Expr(
        :macrocall,
        GlobalRef(Reactant, Symbol("@allowscalar")),
        LineNumberNode(0),
        expr.args[2],
    )
    # Reactant cannot compile the opaque closures RGF creates for @trace.
    return Core.eval(model.mod, expr)
end

# Model arguments should be constant data, with only parameters traced. However,
# untraced arrays make some matmuls very slow, so we convert all model arguments.
# https://github.com/EnzymeAD/Reactant.jl/issues/3167
# https://github.com/EnzymeAD/Reactant.jl/issues/3195
_reactant_arg(value; kwargs...) = Reactant.to_rarray(WrenCore._dealias(value); kwargs...)
_reactant_arg(model::WrenCore.Model; kwargs...) = Reactant.to_rarray(model; kwargs...)

"""
    Reactant.to_rarray(model::Model; kwargs...)

Convert a Wren model's arguments, conditioned values, and model-valued arguments for Reactant.
Accepts the same conversion keywords as `Reactant.to_rarray`.

Use `Reactant.@trace` explicitly for control flow that depends on traced values.
The converted model supports `simulate` directly and is also used by `VecModel`.
"""
function Reactant.to_rarray(model::WrenCore.Model; kwargs...)
    return WrenCore.Model(
        model.name,
        map(value -> _reactant_arg(value; kwargs...), model.args),
        model.stmts,
        model.mod;
        conditioned=map(value -> _reactant_arg(value; kwargs...), model.conditioned),
        enzyme_copy_argnames=Val(keys(model.args)),
        _unsafe_reuse_evaluation_fn=_reactant_evaluation_fn(model),
        _unsafe_skip_dealias=true,
    )
end
