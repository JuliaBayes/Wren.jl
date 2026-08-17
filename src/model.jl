using RuntimeGeneratedFunctions
using VarNames: VarNames, varname, VarName, VarNamedTuple

abstract type ModelExpr end
# TODO: type bounds on L and R probably Union{Expr,Symbol}
struct Sample{L,R} <: ModelExpr
    lhs::L
    rhs::R
end
struct Deterministic{L,R} <: ModelExpr
    lhs::L
    rhs::R
end

struct Model{A<:NamedTuple,F}
    args::A
    # todo: kwargs
    stmts::Vector{Union{ModelExpr,LineNumberNode}}
    # Module that the model was defined in. Required to make RGF resolve globals / functions
    # defined in that module.
    mod::Module
    # Runtime generated evaluation function
    evaluation_fn::F

    function Model(args::A, stmts::Vector{Union{ModelExpr,LineNumberNode}}, mod::Module) where A<:NamedTuple
        evaluation_fn = make_sample_fn(args, stmts, mod)
        new{A,typeof(evaluation_fn)}(args, stmts, mod, evaluation_fn)
    end
end

## output

get_top_level_symbol(expr::Symbol) = expr
function get_top_level_symbol(expr::Expr)
    # TODO(penelopeysm): what about Meta.isexpr(expr, :$)?
    return if Meta.isexpr(expr, :ref)
        get_top_level_symbol(expr.args[1])
    elseif Meta.isexpr(expr, :.)
        get_top_level_symbol(expr.args[1])
    else
        error("unreachable")
    end
end

function make_sample_fn(::NamedTuple{argnames}, stmts::Vector{Union{ModelExpr,LineNumberNode}}, mod::Module) where {argnames}
    body = Expr(:block)
    # insert model arguments at the top of the function body
    @gensym model_args
    for k in argnames
        push!(body.args, :($(k) = $model_args.$k))
    end
    @gensym v
    push!(body.args, :($v = $(VarNamedTuple)()))
    for stmt in stmts
        if stmt isa Sample
            @gensym val vn
            push!(body.args, :($val = $(rand)($(stmt.rhs))))
            push!(body.args, :($(stmt.lhs) = $val))
            # TODO: this line should use BangBang as per DynamicPPL
            push!(body.args, :($vn = $(varname(stmt.lhs, false))))
            push!(body.args, :($v = $(VarNames.templated_setindex!!)($v, $val, $vn, $(get_top_level_symbol(stmt.lhs)))))
        elseif stmt isa Deterministic
            push!(body.args, :($(stmt.lhs) = $(stmt.rhs)))
        end
    end
    push!(body.args, :(return $v))
    return @RuntimeGeneratedFunction(mod, :(function ($model_args,)
        $body
    end))
end

# for debugging
function view_evaluation_fn(m::Model)
    return RuntimeGeneratedFunctions.get_expression(m.evaluation_fn)
end

## macro stuff

function get_function_name_and_args(signature::Expr)
    if signature.head == :call
        return (signature.args[1], signature.args[2:end])
    elseif signature.head == :where
        return get_function_name_and_args(signature.args[1])
    else
        error("@model: expected function signature")
    end
end

to_modelexprs(lnn::LineNumberNode, ::Tuple=()) = [lnn]
function to_modelexprs(expr::Expr, expr_head_stack::Tuple=())
    if expr.head == :call && expr.args[1] == :~
        # lhs ~ rhs
        length(expr.args) == 3 || error("@model: expected ~ to have 2 arguments")
        lhs, rhs = expr.args[2:3]
        return [Sample(lhs, rhs)]
    end

    if expr.head == :(=) && expr.args[1] isa Symbol && !isempty(expr_head_stack) && first(expr_head_stack) != :for
        # lhs = rhs
        length(expr.args) == 2 || error("@model: expected = to have 2 arguments")
        lhs, rhs = expr.args
        return [Deterministic(lhs, rhs)]
    end

    if expr.head == :block
        expr_head_stack = (expr.head, expr_head_stack...)
        exprs = vcat([to_modelexprs(stmt, expr_head_stack) for stmt in expr.args]...)
        # The above infers to `Any`
        return convert(Vector{Union{ModelExpr,LineNumberNode}}, exprs)
    end

    # Something else
    error("todo")
end

function to_namedtuple(args_expr::Expr)
    args_expr.head == :tuple || error("@model: expected a tuple of arguments")
    if isempty(args_expr.args)
        :(NamedTuple())
    else
        new_expr = Expr(:tuple)
        for arg in args_expr.args
            arg isa Symbol || error("@model: expected function argument to be a symbol")
            push!(new_expr.args, :($arg = $(esc(arg))))
        end
        new_expr
    end
end

macro model(expr)
    _model(expr)
end
function _model(expr::Expr)
    # Sanity check
    expr.head == :function || error("@model: expected a function definition")
    length(expr.args) == 2 || error("@model: expected function expr with 2 expr-args")

    # Generate a new symbol for the model's body, since we want the original
    # symbol to directly create the `Model`.
    _, fargs = get_function_name_and_args(expr.args[1])
    fargs_expr = Expr(:tuple, fargs...)
    signature, body = expr.args

    model_args = to_namedtuple(fargs_expr)
    model_exprs = to_modelexprs(body)
    function_body = :($(Model)($model_args, $model_exprs, @__MODULE__))
    function_expr = Expr(:function, esc(signature), function_body)

    return quote
        RuntimeGeneratedFunctions.init(@__MODULE__)
        $function_expr
    end
end

export @model
