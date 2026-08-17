using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

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

struct Model{A<:NamedTuple,C<:VarNamedTuple,F}
    # Arguments passed to the model
    args::A
    # todo: kwargs
    stmts::Vector{Union{ModelExpr,LineNumberNode}}
    # Module that the model was defined in. Required to make RGF resolve globals / functions
    # defined in that module.
    mod::Module
    conditioned::C
    # The cached evaluation function
    evaluation_fn::F
end

function Model(
    args::NamedTuple, stmts::Vector{Union{ModelExpr,LineNumberNode}}, mod::Module
)
    evaluation_fn = make_evaluation_fn(args, stmts, mod)
    return Model(args, stmts, mod, VarNamedTuple(), evaluation_fn)
end

## output

get_top_level_symbol(expr::Symbol) = expr
function get_top_level_symbol(expr::Expr)
    return if Meta.isexpr(expr, :ref)
        get_top_level_symbol(expr.args[1])
    elseif Meta.isexpr(expr, :.)
        get_top_level_symbol(expr.args[1])
    else
        # TODO(penelopeysm): not sure if Meta.isexpr(expr, :$) needs to be handled
        error("unreachable")
    end
end

function make_evaluation_fn(::NamedTuple{argnames}, stmts::Vector{Union{ModelExpr,LineNumberNode}}, mod::Module) where {argnames}
    body = Expr(:block)
    # Insert model arguments at the top of the function body.
    @gensym model_args
    for k in argnames
        push!(body.args, :($(k) = $model_args.$k))
    end

    @gensym accs
    @gensym conditioned
    for stmt in stmts
        if stmt isa Sample
            @gensym rhs val vn
            push!(body.args, :($rhs = $(stmt.rhs))) # evaluate rhs once only
            push!(body.args, :($vn = $(varname(stmt.lhs, false))))
            template = get_top_level_symbol(stmt.lhs)
            push!(body.args, quote
                if $(haskey)($conditioned, $vn)
                    $val = $(getindex)($conditioned, $vn)
                    $(stmt.lhs) = $val
                    $accs = $(accumulate_observe!!)(
                        $accs, $val, $vn, $rhs, $template
                    )
                else
                    $val = $(rand)($rhs)
                    $(stmt.lhs) = $val
                    $accs = $(accumulate_assume!!)(
                        $accs, $val, $vn, $rhs, $template
                    )
                end
            end)
        elseif stmt isa Deterministic
            push!(body.args, :($(stmt.lhs) = $(stmt.rhs)))
        end
    end
    push!(body.args, :(return $accs))
    return @RuntimeGeneratedFunction(mod, :(function ($accs, $model_args, $conditioned)
        $body
    end))
end

# for debugging
function view_evaluation_fn(m::Model)
    return RuntimeGeneratedFunctions.get_expression(m.evaluation_fn)
end

@inline function evaluate(model::Model, accs::AccumulatorTuple)
    return model.evaluation_fn(accs, model.args, model.conditioned)
end

conditioned(model::Model) = model.conditioned

function condition(model::Model, values::VarNamedTuple)
    new_conditioned = merge(model.conditioned, values)
    return Model(model.args, model.stmts, model.mod, new_conditioned, model.evaluation_fn)
end
Base.:|(model::Model, values::VarNamedTuple) = condition(model, values)

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
