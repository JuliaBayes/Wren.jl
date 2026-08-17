using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

abstract type ModelExpr end
# TODO: type bounds on L and R probably Union{Expr,Symbol}
struct Sample{L,R} <: ModelExpr
    lhs::L
    rhs::R
end
struct Observe{L,R} <: ModelExpr
    lhs::L
    rhs::R
end
struct ReturnStatement{E} <: ModelExpr
    expr::E
end
struct OtherStatement{E} <: ModelExpr
    expr::E
end
struct JuliaExpr{A} <: ModelExpr
    head::Symbol
    args::A
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

# VarNames.varname(:(xs[i])) returns something like
#     VarName{:xs}(Index((esc(i),), ...))
# which gets `i` from the caller's scope. However, when we insert this expression into a
# larger Expr (as in `@model` and its RGF), we want to take `i` from the scope of the larger
# Expr itself, rather than the caller's scope. That necessitates removing the `esc()` call.
remove_escapes(x) = x
function remove_escapes(expr::Expr)
    expr.head == :escape && return remove_escapes(only(expr.args))
    return Expr(expr.head, map(remove_escapes, expr.args)...)
end

emit_model_expr(expr, accs, conditioned, latent_source) = expr
function emit_model_expr(stmt::JuliaExpr, accs, conditioned, latent_source)
    args = map(
        arg -> emit_model_expr(arg, accs, conditioned, latent_source), stmt.args
    )
    return Expr(stmt.head, args...)
end
emit_model_expr(stmt::OtherStatement, accs, conditioned, latent_source) = stmt.expr

function emit_model_expr(stmt::Sample, accs, conditioned, latent_source)
    @gensym rhs val vn
    rhs_expr = emit_model_expr(stmt.rhs, accs, conditioned, latent_source)
    vn_expr = remove_escapes(varname(stmt.lhs, false))
    template = get_top_level_symbol(stmt.lhs)
    return quote
        $rhs = $rhs_expr
        $vn = $vn_expr
        if $(haskey)($conditioned, $vn)
            $val = $(getindex)($conditioned, $vn)
            $(stmt.lhs) = $val
            $accs = $(accumulate_observe!!)($accs, $val, $vn, $rhs, $template)
        else
            $val = $(get_latent)($latent_source, $vn, $rhs)
            $(stmt.lhs) = $val
            $accs = $(accumulate_assume!!)($accs, $val, $vn, $rhs, $template)
        end
        # `x ~ dist` evaluates to the value of x
        $val
    end
end

function emit_model_expr(stmt::Observe, accs, conditioned, latent_source)
    @gensym rhs val vn
    rhs_expr = emit_model_expr(stmt.rhs, accs, conditioned, latent_source)
    vn_expr = remove_escapes(varname(stmt.lhs, false))
    template = get_top_level_symbol(stmt.lhs)
    return quote
        $val = $(stmt.lhs)
        $rhs = $rhs_expr
        $vn = $vn_expr
        $accs = $(accumulate_observe!!)($accs, $val, $vn, $rhs, $template)
    end
end

function emit_model_expr(stmt::ReturnStatement, accs, conditioned, latent_source)
    body = Expr(:block)
    if stmt.expr !== nothing
        push!(body.args, emit_model_expr(stmt.expr, accs, conditioned, latent_source))
    end
    push!(body.args, :(return $accs))
    return body
end

function emit_statements(stmts, accs, conditioned, latent_source)
    body = Expr(:block)
    for stmt in stmts
        push!(body.args, emit_model_expr(stmt, accs, conditioned, latent_source))
    end
    return body
end

function make_evaluation_fn(::NamedTuple{argnames}, stmts::Vector{Union{ModelExpr,LineNumberNode}}, mod::Module) where {argnames}
    body = Expr(:block)
    @gensym model_args
    for k in argnames
        push!(body.args, :($(k) = $model_args.$k))
    end

    @gensym accs conditioned latent_source
    append!(body.args, emit_statements(stmts, accs, conditioned, latent_source).args)
    push!(body.args, :(return $accs))
    return @RuntimeGeneratedFunction(mod, :(function (
        $latent_source, $accs, $model_args, $conditioned
    )
        $body
    end))
end

# for debugging
function view_evaluation_fn(m::Model)
    return RuntimeGeneratedFunctions.get_expression(m.evaluation_fn)
end

@inline function evaluate(
    model::Model,
    latent_source::AbstractLatentSource,
    accs::AccumulatorTuple,
)
    return model.evaluation_fn(latent_source, accs, model.args, model.conditioned)
end

function Base.rand(rng::AbstractRNG, model::Model)
    accs = evaluate(model, FromPrior(rng), AccumulatorTuple(RawValueAccumulator()))
    return get_raw_values(accs)
end

Base.rand(model::Model) = rand(default_rng(), model)

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

function to_modelexprs(expr, mod::Module)
    dest = Union{ModelExpr,LineNumberNode}[]
    if Meta.isexpr(expr, :block)
        for stmt in expr.args
            to_modelexprs!(dest, stmt, mod)
        end
    else
        to_modelexprs!(dest, expr, mod)
    end
    return dest
end

function to_modelexprs!(dest, lnn::LineNumberNode, ::Module)
    push!(dest, lnn)
    return dest
end

function to_modelexprs!(dest, expr, mod::Module)
    lowered = lower_model_expr(expr, mod, true)
    push!(dest, lowered isa ModelExpr ? lowered : OtherStatement(lowered))
    return dest
end

lower_model_expr(expr, ::Module, ::Bool=false) = expr

function lower_model_expr(expr::Expr, mod::Module, statement_position::Bool=false)
    if expr.head == :macrocall && expr.args[1] == Symbol("@observe")
        statement_position || error("@observe must be used as a statement")
        observed_expr = last(expr.args)
        if !(observed_expr isa Expr && observed_expr.head == :call && observed_expr.args[1] == :~)
            error("@observe: expected an expression of the form `value ~ distribution`")
        end
        length(observed_expr.args) == 3 ||
            error("@observe: expected ~ to have 2 arguments")
        lhs, rhs = observed_expr.args[2:3]
        return Observe(lhs, lower_model_expr(rhs, mod))
    end

    if expr.head == :macrocall
        expanded = macroexpand(mod, expr; recursive=false)
        return lower_model_expr(expanded, mod, statement_position)
    end

    if expr.head == :call && expr.args[1] == :~
        length(expr.args) == 3 || error("@model: expected ~ to have 2 arguments")
        lhs, rhs = expr.args[2:3]
        return Sample(lhs, lower_model_expr(rhs, mod))
    end

    if expr.head == :return
        value = only(expr.args)
        return ReturnStatement(value === nothing ? nothing : lower_model_expr(value, mod))
    end

    if expr.head in (:quote, :function, :(->), :macro, :module, :baremodule)
        return OtherStatement(expr)
    end

    args = map(enumerate(expr.args)) do (index, arg)
        child_is_statement = expr.head in (:block, :toplevel) ||
                             expr.head in (:if, :elseif) && index >= 2 ||
                             expr.head in (:while, :for) && index == 2 ||
                             expr.head == :let && index == length(expr.args)
        lower_model_expr(arg, mod, child_is_statement)
    end
    return JuliaExpr(expr.head, args)
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
    _model(expr, __module__)
end

macro observe(expr)
    error("@observe must be used inside @model")
end

function _model(expr::Expr, mod::Module)
    # Sanity check
    expr.head == :function || error("@model: expected a function definition")
    length(expr.args) == 2 || error("@model: expected function expr with 2 expr-args")

    # Generate a new symbol for the model's body, since we want the original
    # symbol to directly create the `Model`.
    _, fargs = get_function_name_and_args(expr.args[1])
    fargs_expr = Expr(:tuple, fargs...)
    signature, body = expr.args

    model_args = to_namedtuple(fargs_expr)
    model_exprs = to_modelexprs(body, mod)
    function_body = :($(Model)($model_args, $model_exprs, @__MODULE__))
    function_expr = Expr(:function, esc(signature), function_body)

    return quote
        RuntimeGeneratedFunctions.init(@__MODULE__)
        $function_expr
    end
end
