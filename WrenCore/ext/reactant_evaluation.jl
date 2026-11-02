using WrenCore.Expressions: AbstractWrenExpr, WrenExpr, OtherStatement, ReturnStatement

_control_transfer(x) = false
_control_transfer(::ReturnStatement) = true
_control_transfer(x::OtherStatement) = _control_transfer(x.expr)
_control_transfer(x::WrenExpr) =
    x.head in (:return, :break, :continue) || any(_control_transfer, x.args)
_control_transfer(x::Expr) =
    x.head in (:return, :break, :continue) ||
    (!(x.head in (:quote, :function, :->)) && any(_control_transfer, x.args))

_trace_model_expr(x, value_needed=true) = x
function _trace_model_expr(x::AbstractWrenExpr, value_needed=true)
    constructor = Base.typename(typeof(x)).wrapper
    return constructor(
        map(name -> _trace_model_expr(getfield(x, name)), fieldnames(typeof(x)))...,
    )
end
_trace_model_expr(x::OtherStatement, value_needed=true) = x
function _trace_model_expr(x::WrenExpr, value_needed=true)
    args = map(enumerate(x.args)) do (i, arg)
        child_value = if x.head == :block
            value_needed && i == length(x.args)
        elseif x.head in (:if, :elseif)
            i == 1 || value_needed
        elseif x.head in (:for, :while)
            i == 1
        else
            true
        end
        _trace_model_expr(arg, child_value)
    end
    expr = WrenExpr(x.head, args)
    # Static early returns and loop exits remain native. Reactant cannot trace them.
    if x.head == :if && !_control_transfer(x)
        if !value_needed
            return WrenExpr(
                :macrocall,
                Any[GlobalRef(Reactant, Symbol("@trace")), LineNumberNode(0), expr],
            )
        end
        # Give expression-valued conditionals the assignment form of @trace.
        value = Symbol("wren_branch_value_", replace(string(gensym()), '#' => '_'))
        assignment = WrenExpr(:(=), Any[value, expr])
        traced = WrenExpr(
            :macrocall,
            Any[GlobalRef(Reactant, Symbol("@trace")), LineNumberNode(0), assignment],
        )
        return WrenExpr(:block, Any[traced, value])
    end
    return expr
end

function _traced_evaluation_fn(model::WrenCore.Model, return_value=true)
    last_statement = findlast(stmt -> !(stmt isa LineNumberNode), model.stmts)
    stmts = Union{AbstractWrenExpr,LineNumberNode}[
        _trace_model_expr(stmt, return_value && i == last_statement) for
        (i, stmt) in enumerate(model.stmts)
    ]
    # Reactant's branch dataflow intentionally ignores gensyms, including Wren's
    # accumulator argument unless we give it an ordinary local identifier.
    fn_args = ntuple(_ -> Symbol("wren_local_", replace(string(gensym()), '#' => '_')), 7)
    expr = WrenCore.evaluation_fn_expr(keys(model.args), stmts; fn_args)
    conditioned = fn_args[5]
    pushfirst!(
        expr.args[2].args,
        :($conditioned = $(map)($(WrenCore._dealias), $conditioned)),
    )
    expr.args[2] = Expr(
        :macrocall,
        GlobalRef(Reactant, Symbol("@allowscalar")),
        LineNumberNode(0),
        expr.args[2],
    )
    return Core.eval(model.mod, expr)
end

_reactant_model(value) = value
_reactant_conditioned(value) = WrenCore._dealias(value)
_reactant_conditioned(model::WrenCore.Model) = _reactant_model(model)
function _reactant_model(model::WrenCore.Model)
    return WrenCore.Model(
        model.name,
        map(_reactant_model, model.args),
        model.stmts,
        model.mod;
        conditioned=map(_reactant_conditioned, model.conditioned),
        enzyme_copy_argnames=Val(keys(model.args)),
        _unsafe_reuse_evaluation_fn=_traced_evaluation_fn(model),
    )
end
