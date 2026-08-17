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
struct Deterministic{L,R} <: ModelExpr
    lhs::L
    rhs::R
end
struct ReturnStatement{E} <: ModelExpr
    expr::E
end
struct OtherStatement{E} <: ModelExpr
    expr::E
end
struct IfStatement{C,T,E} <: ModelExpr
    condition::C
    then_branch::T
    else_branch::E
end
struct WhileStatement{C,B} <: ModelExpr
    condition::C
    body::B
end
struct ForStatement{I,B} <: ModelExpr
    iterator::I
    body::B
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

function emit_statements(stmts, accs, conditioned, latent_source)
    body = Expr(:block)
    for stmt in stmts
        if stmt isa Sample
            @gensym rhs val vn
            push!(body.args, :($rhs = $(stmt.rhs))) # evaluate rhs once only
            vn_expr = remove_escapes(varname(stmt.lhs, false))
            push!(body.args, :($vn = $vn_expr))
            template = get_top_level_symbol(stmt.lhs)
            push!(body.args, quote
                if $(haskey)($conditioned, $vn)
                    $val = $(getindex)($conditioned, $vn)
                    $(stmt.lhs) = $val
                    $accs = $(accumulate_observe!!)(
                        $accs, $val, $vn, $rhs, $template
                    )
                else
                    $val = $(get_latent)($latent_source, $vn, $rhs)
                    $(stmt.lhs) = $val
                    $accs = $(accumulate_assume!!)(
                        $accs, $val, $vn, $rhs, $template
                    )
                end
            end)
        elseif stmt isa Observe
            @gensym rhs val vn
            push!(body.args, :($val = $(stmt.lhs)))
            push!(body.args, :($rhs = $(stmt.rhs)))
            vn_expr = remove_escapes(varname(stmt.lhs, false))
            push!(body.args, :($vn = $vn_expr))
            template = get_top_level_symbol(stmt.lhs)
            push!(body.args, :($accs = $(accumulate_observe!!)(
                $accs, $val, $vn, $rhs, $template
            )))
        elseif stmt isa Deterministic
            push!(body.args, :($(stmt.lhs) = $(stmt.rhs)))
        elseif stmt isa ReturnStatement
            # need to evaluate the original return value in case it has
            # side effects.
            stmt.expr !== nothing && push!(body.args, stmt.expr)
            # but we don't actually return it -- only the accs
            push!(body.args, :(return $accs))
        elseif stmt isa OtherStatement
            push!(body.args, stmt.expr)
        elseif stmt isa IfStatement
            push!(body.args, emit_if(stmt, accs, conditioned, latent_source))
        elseif stmt isa WhileStatement
            loop_body = emit_statements(stmt.body, accs, conditioned, latent_source)
            push!(body.args, Expr(:while, stmt.condition, loop_body))
        elseif stmt isa ForStatement
            loop_body = emit_statements(stmt.body, accs, conditioned, latent_source)
            push!(body.args, Expr(:for, stmt.iterator, loop_body))
        elseif stmt isa LineNumberNode
            push!(body.args, stmt)
        end
    end
    return body
end

function emit_if(stmt::IfStatement, accs, conditioned, latent_source)
    then_branch = emit_statements(stmt.then_branch, accs, conditioned, latent_source)
    if stmt.else_branch === nothing
        return Expr(:if, stmt.condition, then_branch)
    end
    else_branch = stmt.else_branch isa IfStatement ?
        emit_if(stmt.else_branch, accs, conditioned, latent_source) :
        emit_statements(stmt.else_branch, accs, conditioned, latent_source)
    return Expr(:if, stmt.condition, then_branch, else_branch)
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

function to_modelexprs(expr, expr_head_stack::Tuple=())
    dest = Union{ModelExpr,LineNumberNode}[]
    return to_modelexprs!(dest, expr, expr_head_stack)
end

function to_modelexprs!(dest, lnn::LineNumberNode, ::Tuple)
    push!(dest, lnn)
    return dest
end

function to_if_statement(expr::Expr, expr_head_stack::Tuple)
    condition = expr.args[1]
    then_branch = to_modelexprs(expr.args[2], (:if, expr_head_stack...))
    else_branch = if length(expr.args) == 2
        nothing
    elseif Meta.isexpr(expr.args[3], :elseif)
        to_if_statement(expr.args[3], expr_head_stack)
    else
        to_modelexprs(expr.args[3], (:if, expr_head_stack...))
    end
    return IfStatement(condition, then_branch, else_branch)
end

function to_modelexprs!(dest, expr::Expr, expr_head_stack::Tuple)
    if expr.head == :macrocall && expr.args[1] == Symbol("@observe")
        observed_expr = last(expr.args)
        if !(observed_expr isa Expr && observed_expr.head == :call && observed_expr.args[1] == :~)
            error("@observe: expected an expression of the form `value ~ distribution`")
        end
        length(observed_expr.args) == 3 ||
            error("@observe: expected ~ to have 2 arguments")
        lhs, rhs = observed_expr.args[2:3]
        push!(dest, Observe(lhs, rhs))
        return dest
    end

    if expr.head == :call && expr.args[1] == :~
        # lhs ~ rhs
        length(expr.args) == 3 || error("@model: expected ~ to have 2 arguments")
        lhs, rhs = expr.args[2:3]
        push!(dest, Sample(lhs, rhs))
        return dest
    end

    if expr.head == :(=) && expr.args[1] isa Symbol && !isempty(expr_head_stack) && first(expr_head_stack) != :for
        # lhs = rhs
        length(expr.args) == 2 || error("@model: expected = to have 2 arguments")
        lhs, rhs = expr.args
        push!(dest, Deterministic(lhs, rhs))
        return dest
    end

    if expr.head == :return
        push!(dest, ReturnStatement(only(expr.args)))
        return dest
    end

    if expr.head == :if
        push!(dest, to_if_statement(expr, expr_head_stack))
        return dest
    end

    if expr.head == :while
        condition, body = expr.args
        push!(dest, WhileStatement(
            condition, to_modelexprs(body, (:while, expr_head_stack...))
        ))
        return dest
    end

    if expr.head == :for
        iterator, body = expr.args
        push!(dest, ForStatement(
            iterator, to_modelexprs(body, (:for, expr_head_stack...))
        ))
        return dest
    end

    if expr.head == :block
        expr_head_stack = (expr.head, expr_head_stack...)
        for stmt in expr.args
            to_modelexprs!(dest, stmt, expr_head_stack)
        end
        return dest
    end

    push!(dest, OtherStatement(expr))
    return dest
end

function to_modelexprs!(dest, expr, ::Tuple)
    push!(dest, OtherStatement(expr))
    return dest
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

macro observe(expr)
    error("@observe must be used inside @model")
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
