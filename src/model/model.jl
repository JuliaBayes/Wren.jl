# module Wren

using Accessors: Accessors
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

"""
    Wren.Model

A Wren model.

$(TYPEDFIELDS)
"""
struct Model{A<:NamedTuple,C<:VarNamedTuple,F}
    "Name of the function declared with @wren"
    name::Symbol
    "Arguments passed to the model. TODO(penelopeysm): kwargs"
    args::A
    "The model's AST, stored as Wren's probabilistic IR"
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}}
    "Module that the model was defined in. Required to make RGF resolve globals / functions
    defined in that module."
    mod::Module
    "The set of variables that the model is conditioned on."
    conditioned::C
    "The cached evaluation function."
    evaluation_fn::F

    function Model(
        name::Symbol,
        args::NamedTuple,
        stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
        mod::Module;
        conditioned::VarNamedTuple=VarNamedTuple(),
    )
        evaluation_fn = make_evaluation_fn(args, stmts, mod)
        return new{typeof(args),typeof(conditioned),typeof(evaluation_fn)}(
            name,
            args,
            stmts,
            mod,
            conditioned,
            evaluation_fn,
        )
    end
end

function make_evaluation_fn(
    ::NamedTuple{argnames},
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    mod::Module,
) where {argnames}
    body = Expr(:block)
    @gensym model_args
    for k in argnames
        push!(body.args, :($(k) = $model_args.$k))
    end

    @gensym accs conditioned latent_source parameter_space retval
    model_body = emit_statements(stmts, accs, conditioned, latent_source, parameter_space)
    push!(body.args, :($retval = $model_body))
    push!(body.args, :(return ($retval, $accs)))
    return @RuntimeGeneratedFunction(
        mod,
        :(function ($latent_source, $parameter_space, $accs, $model_args, $conditioned)
            $body
        end)
    )
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
        error("@wren: expected function signature")
    end
end

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    println(io, "Model ", model.name)
    print(io, "  arguments: ")
    show(io, model.args)
    print(io, "\n  conditioned: ")
    show(io, model.conditioned)
    println(io, "\n  expressions:")
    return _show_wren_exprs(io, model.stmts, 4)
end

function to_namedtuple(args_expr::Expr)
    args_expr.head == :tuple || error("@wren: expected a tuple of arguments")
    if isempty(args_expr.args)
        :(NamedTuple())
    else
        new_expr = Expr(:tuple)
        for arg in args_expr.args
            arg isa Symbol || error("@wren: expected function argument to be a symbol")
            push!(new_expr.args, :($arg = $(esc(arg))))
        end
        new_expr
    end
end

macro wren(expr)
    _wren(expr, __module__)
end

macro observe(expr)
    error("@observe must be used inside @wren")
end

macro plate(args...)
    error("@plate must be used inside @wren")
end

function _wren(expr::Expr, mod::Module)
    # Sanity check
    expr.head == :function || error("@wren: expected a function definition")
    length(expr.args) == 2 || error("@wren: expected function expr with 2 expr-args")

    # Generate a new symbol for the model's body, since we want the original
    # symbol to directly create the `Model`.
    fname, fargs = get_function_name_and_args(expr.args[1])
    fname isa Symbol || error("@wren: expected function name to be a symbol")
    fargs_expr = Expr(:tuple, fargs...)
    signature, body = expr.args

    model_args = to_namedtuple(fargs_expr)
    model_exprs = to_wren_exprs(body, mod)
    function_body = :($(Model)($(QuoteNode(fname)), $model_args, $model_exprs, @__MODULE__))
    function_expr = Expr(:function, esc(signature), function_body)

    return quote
        RuntimeGeneratedFunctions.init(@__MODULE__)
        $function_expr
    end
end
