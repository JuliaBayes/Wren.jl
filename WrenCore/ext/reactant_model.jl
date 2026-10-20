# Branch functions must be separate RGFs: generated model bodies cannot define closures.
_reactant_branch_output(x) =
    throw(ArgumentError("Reactant model branches cannot return untraced custom state"))
_reactant_branch_output(x::Nothing) = x
_reactant_branch_output(x::Reactant.ReactantCore.MissingTracedValue) = x
_reactant_branch_output(x::AbstractArray{<:Number}) =
    Reactant.call_with_native(Reactant.promote_to, Reactant.TracedRArray, x)
_reactant_branch_output(x::WrenCore.AliasAccumulator) = x
_reactant_branch_output(x::Number) = Reactant.ReactantCore.promote_to_traced(x)
_reactant_branch_output(x::Union{Tuple,NamedTuple,WrenCore.AccumulatorTuple}) =
    map(_reactant_branch_output, x)
_reactant_branch_output(x::WrenCore.LogProbAccumulator) =
    WrenCore.basetypeof(x)(_reactant_branch_output(WrenCore.logp(x)))
_reactant_branch_output(x::WrenCore.NoLogProb) = x

function _reactant_check_branch_input(x, snapshots=Tuple[])
    if x isa Union{Tuple,NamedTuple}
        foreach(value -> _reactant_check_branch_input(value, snapshots), x)
        return snapshots
    end
    x isa AbstractArray{<:Reactant.ReactantPrimitive} && push!(snapshots, (x, copy(x)))
    x isa Function && isbitstype(typeof(x)) && return snapshots
    x isa Union{
        Number,
        AbstractArray{<:Number},
        Nothing,
        Symbol,
        Module,
        Type,
        WrenCore.FromVector,
        WrenCore.AbstractParameterSpace,
        WrenCore.AccumulatorTuple,
        WrenCore.EvalContext,
        WrenCore.VarNamedTuple,
        WrenCore.VarName,
        WrenCore.Model,
        Reactant.ReactantCore.MissingTracedValue,
    } && return snapshots
    throw(ArgumentError("Reactant model branches cannot capture untraced custom state"))
end

# Read-only host indices stay static. Reject writes before the other arm can
# observe them; this check runs while compiling, not in the compiled density.
function _reactant_check_branch_mutation(snapshots)
    changed = false
    for (value, original) in snapshots
        if !isequal(value, original)
            copyto!(value, original)
            changed = true
        end
    end
    changed && throw(ArgumentError("Reactant model branches cannot mutate host arrays"))
    return nothing
end

struct ReactantModelBranch{F}
    f::F
end
function (branch::ReactantModelBranch)(args)
    snapshots = Reactant.call_with_native(_reactant_check_branch_input, args)
    result = _reactant_branch_output(branch.f(args))
    Reactant.call_with_native(_reactant_check_branch_mutation, snapshots)
    return result
end
_reactant_traced_if(cond, left, right, args; track_numbers) =
    cond ? left(args) : right(args)
function _reactant_traced_if(
    cond::Reactant.TracedRNumber{Bool},
    left,
    right,
    args;
    track_numbers,
)
    return Reactant.ReactantCore.traced_if(
        cond,
        ReactantModelBranch(left),
        ReactantModelBranch(right),
        args;
        track_numbers,
    )
end

function _reactant_branch_functions!(names, expr)
    expr isa Expr || return names
    if expr.head === :call && expr.args[1] === Reactant.ReactantCore.traced_if
        union!(names, expr.args[(end-2):(end-1)])
    end
    foreach(arg -> _reactant_branch_functions!(names, arg), expr.args)
    return names
end

function _reactant_hoist_branches(expr, names, mod)
    expr isa Expr || return expr
    expr =
        Expr(expr.head, map(arg -> _reactant_hoist_branches(arg, names, mod), expr.args)...)
    if expr.head === :call && expr.args[1] === Reactant.ReactantCore.traced_if
        expr.args[1] = _reactant_traced_if
    end
    if expr.head === :(=) && expr.args[1] in names && Meta.isexpr(expr.args[2], :(->))
        fn = expr.args[2]
        expr.args[2] = QuoteNode(
            WrenCore.RuntimeGeneratedFunctions.RuntimeGeneratedFunction(mod, mod, fn),
        )
    end
    return expr
end

struct ReactantModelEmitter
    mod::Module
end

_reactant_trace_names(x, names) = x
_reactant_trace_names(x::Symbol, names) =
    startswith(string(x), '#') ? get!(() -> Symbol("_wren_", gensym(:state)), names, x) : x
_reactant_trace_names(x::Expr, names) =
    x.head === :quote ? x :
    Expr(x.head, map(arg -> _reactant_trace_names(arg, names), x.args)...)

function (emitter::ReactantModelEmitter)(stmt::WrenCore.Observe, accs, args...)
    body = WrenCore.emit_model_expr(stmt, accs, args...)
    return :($accs = let $accs = $accs
        $body
        $accs
    end)
end

(emitter::ReactantModelEmitter)(stmt, args...) = WrenCore.emit_model_expr(stmt, args...)

function (emitter::ReactantModelEmitter)(stmt::WrenCore.Sample, accs, args...)
    @gensym retval
    root = WrenCore.get_top_level_symbol(stmt.lhs)
    sample = WrenCore.emit_model_expr(stmt, accs, args...)
    bindings = Expr(:block, :($accs = $accs))
    stmt.lhs isa Symbol || push!(bindings.args, :($root = $root))
    # Export the tilde's value, updated root and accumulators; keep transform and
    # distribution temporaries local so Reactant does not try to merge them.
    return quote
        ($retval, $root, $accs) = $(Expr(:let, bindings, :(($sample, $root, $accs))))
        $retval
    end
end

function (emitter::ReactantModelEmitter)(stmt::WrenCore.WrenExpr, args...)
    stmt.head === :if ||
        return Expr(stmt.head, map(arg -> emitter(arg, args...), stmt.args)...)
    # Reactant drops gensyms from branch inputs. Keep Wren's state visible to tracing.
    names = Dict{Symbol,Symbol}()
    inputs = map(arg -> _reactant_trace_names(arg, names), args)
    bindings = [:($input = $arg) for (input, arg) in zip(inputs, args) if input !== arg]
    expr = Expr(stmt.head, map(arg -> emitter(arg, inputs...), stmt.args)...)
    state = Reactant.ReactantCore.ExpressionExplorer.compute_symbols_state(expr)
    generated = filter(
        name -> startswith(string(name), '#'),
        union(state.references, state.assignments),
    )
    for name in generated
        input = _reactant_trace_names(name, names)
        push!(
            bindings,
            :(
                $input =
                    @isdefined($name) ? $name :
                    $(Reactant.ReactantCore.MissingTracedValue)()
            ),
        )
    end
    updates = [
        :($name = $(_reactant_trace_names(name, names))) for
        name in intersect(generated, state.assignments)
    ]
    expr = _reactant_trace_names(expr, names)
    traced = macroexpand(
        emitter.mod,
        Expr(
            :macrocall,
            GlobalRef(Reactant, Symbol("@trace")),
            LineNumberNode(0),
            :(track_numbers=false),
            expr,
        ),
    )
    branches = _reactant_branch_functions!(Set{Symbol}(), traced)
    body = _reactant_hoist_branches(traced, branches, emitter.mod)
    @gensym value
    return quote
        $(bindings...)
        $value = $body
        $(args[1]) = $(inputs[1])
        $(updates...)
        $value
    end
end

_reactant_has_branch(::Any) = false
_reactant_has_branch(stmt::WrenCore.WrenExpr) =
    stmt.head === :if || any(_reactant_has_branch, stmt.args)
function _reactant_evaluation_fn(model)
    any(_reactant_has_branch, model.stmts) || return model.evaluation_fn
    return WrenCore.make_evaluation_fn(
        keys(model.args),
        model.stmts,
        model.mod;
        emitter=ReactantModelEmitter(model.mod),
    )
end

# Ordinary integers remain static under @trace, including explicit alias counts.
# Prove equal counts in every arm before allowing Reactant to merge that state.
_reactant_alias_counts(stmt, groups) =
    WrenCore._alias_straight_expr(stmt) ? map(_ -> 0, groups) : nothing
function _reactant_alias_counts(stmt::WrenCore.Sample, groups)
    WrenCore._alias_straight_expr(stmt.rhs) || return nothing
    return map(groups) do group
        Int(stmt.lhs isa Symbol && WrenCore.VarName{stmt.lhs}() in group)
    end
end
_reactant_alias_counts(stmt::WrenCore.IID, groups) =
    WrenCore._alias_straight_expr(stmt.size) ? _reactant_alias_counts(stmt.tilde, groups) :
    nothing
_reactant_alias_counts(stmt::WrenCore.Observe, groups) =
    WrenCore._alias_straight_expr(stmt.lhs) && WrenCore._alias_straight_expr(stmt.rhs) ?
    map(_ -> 0, groups) : nothing
_reactant_alias_counts(stmt::WrenCore.AddLogLikelihood, groups) =
    _reactant_alias_counts(stmt.expr, groups)
function _reactant_alias_counts(stmt::WrenCore.WrenExpr, groups)
    if stmt.head in (:if, :elseif)
        WrenCore._alias_straight_expr(stmt.args[1]) || return nothing
        left = _reactant_alias_counts(stmt.args[2], groups)
        right =
            length(stmt.args) == 3 ? _reactant_alias_counts(stmt.args[3], groups) :
            map(_ -> 0, groups)
        return isequal(left, right) ? left : nothing
    elseif stmt.head === :block
        counts = map(_ -> 0, groups)
        for arg in stmt.args
            next = _reactant_alias_counts(arg, groups)
            next === nothing && return nothing
            counts = map(+, counts, next)
        end
        return counts
    end
    return WrenCore._alias_straight_expr(stmt) ? map(_ -> 0, groups) : nothing
end

function _reactant_check_aliases(model, vecinfo)
    vecinfo.alias_check === nothing && return nothing
    stmts = filter(stmt -> !(stmt isa LineNumberNode), model.stmts)
    if !isempty(stmts) && last(stmts) isa WrenCore.ReturnStatement
        WrenCore._alias_straight_expr(last(stmts).expr) || throw(
            ArgumentError(
                "Reactant requires explicit aliases to occur once in every straight-line branch",
            ),
        )
        pop!(stmts)
    end
    counts = _reactant_alias_counts(WrenCore.WrenExpr(:block, stmts), vecinfo.aliases)
    counts == map(_ -> 1, vecinfo.aliases) || throw(
        ArgumentError(
            "Reactant requires explicit aliases to occur once in every straight-line branch",
        ),
    )
    return nothing
end
