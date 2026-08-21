using Accessors: Accessors
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

struct Model{A<:NamedTuple,C<:VarNamedTuple,F}
    # Name of the function declared with @wren
    name::Symbol
    # Arguments passed to the model
    args::A
    # todo: kwargs
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}}
    # Module that the model was defined in. Required to make RGF resolve globals / functions
    # defined in that module.
    mod::Module
    conditioned::C
    # The cached evaluation function
    evaluation_fn::F
end

function Model(
    name::Symbol,
    args::NamedTuple,
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    mod::Module,
)
    evaluation_fn = make_evaluation_fn(args, stmts, mod)
    return Model(name, args, stmts, mod, VarNamedTuple(), evaluation_fn)
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
# larger Expr (as in `@wren` and its RGF), we want to take `i` from the scope of the larger
# Expr itself, rather than the caller's scope. That necessitates removing the `esc()` call.
remove_escapes(x) = x
function remove_escapes(expr::Expr)
    expr.head == :escape && return remove_escapes(only(expr.args))
    return Expr(expr.head, map(remove_escapes, expr.args)...)
end

function evaluate_submodel(rhs, latent_source, parameter_space, accs, vn)
    child_source = scope(latent_source, vn)
    child_space = scope(parameter_space, vn)
    child_accs = map(split, accs)
    had_params = haskey(child_accs, Val(PARAMS_ACCNAME))
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

assign_or_set!!(lhs::Symbol, rhs, vn) = :($lhs = $rhs)
function assign_or_set!!(lhs::Expr, rhs, vn)
    left_top_sym = get_top_level_symbol(lhs)
    return :(
        $left_top_sym = $(Accessors.set)(
            $left_top_sym,
            $(VarNames.with_mutation)($(VarNames.getoptic)($vn)),
            $rhs,
        )
    )
end

emit_model_expr(expr, accs, conditioned, latent_source, parameter_space) = expr
function emit_model_expr(stmt::WrenExpr, accs, conditioned, latent_source, parameter_space)
    args = map(
        arg -> emit_model_expr(arg, accs, conditioned, latent_source, parameter_space),
        stmt.args,
    )
    return Expr(stmt.head, args...)
end
function emit_model_expr(
    stmt::OtherStatement,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    return stmt.expr
end

function emit_model_expr(stmt::Sample, accs, conditioned, latent_source, parameter_space)
    @gensym rhs val tval init_tval logjac vn child_accs child_params
    rhs_expr = emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space)
    vn_expr = remove_escapes(varname(stmt.lhs, false))
    template = get_top_level_symbol(stmt.lhs)
    return quote
        $rhs = $rhs_expr
        $vn = $vn_expr
        if $rhs isa $(Model)
            $val, $child_params, $child_accs =
                $(evaluate_submodel)($rhs, $latent_source, $parameter_space, $accs, $vn)
            $(assign_or_set!!(stmt.lhs, child_params, vn))
            $accs = $(combine_with_namespace)(
                $accs,
                $child_accs,
                $vn,
                $(stmt.lhs isa Symbol ? nothing : template),
            )
        else
            if $(haskey)($conditioned, $vn)
                $val = $(getindex)($conditioned, $vn)
                $(assign_or_set!!(stmt.lhs, val, vn))
                $accs = $(accumulate_observe!!)($accs, $val, $vn, $rhs, $template)
            else
                $init_tval = $(get_latent)($latent_source, $vn, $rhs)
                $val, $tval, $logjac =
                    $(apply_parameter_space)($parameter_space, $init_tval, $vn, $rhs)
                $(assign_or_set!!(stmt.lhs, val, vn))
                $accs = $(accumulate_assume!!)(
                    $accs,
                    $val,
                    $tval,
                    $vn,
                    $rhs,
                    $logjac,
                    $template,
                )
            end
        end
        # A tilde expression evaluates to the sampled value or submodel return value.
        $val
    end
end

# Overloaded per distribution backend
function _product end
# @plate for i in 1:N
#    y[i] ~ Normal(x[i])
# end
#
# CURRENTLY gets lowered to something like:
#
# y[1:N] ~ _product([Normal(x[i]) for i in 1:N])
#
# This can be improved! https://github.com/JuliaBayes/ProbabilityMeasures.jl/issues/15
function emit_model_expr(stmt::Plate, accs, conditioned, latent_source, parameter_space)
    # stmt.body isa block expr
    body_exprs = [
        vectorise_plate_stmt(
            s,
            stmt.loop_idx,
            stmt.loop_iters,
            accs,
            conditioned,
            latent_source,
            parameter_space,
        ) for s in stmt.body.args
    ]
    return Expr(:block, body_exprs...)
end
function replace_loop_idx_with_iters(expr, loop_idx::Symbol, loop_iters)
    # replace all occurrences of `i` with `1:N` (for example)
    return if expr isa Symbol
        expr == loop_idx ? loop_iters : expr
    elseif expr isa QuoteNode
        QuoteNode(replace_loop_idx_with_iters(expr.value, loop_idx, loop_iters))
    elseif expr isa Expr
        if expr.head in (:ref, :., :call)
            new_args = map(
                arg -> replace_loop_idx_with_iters(arg, loop_idx, loop_iters),
                expr.args,
            )
            new_expr = Expr(expr.head, new_args...)
            if expr.head == :call && any(arg -> arg == loop_idx, expr.args)
                # e.g. i + 1 --> apart from just replacing i, we also need to broadcast the
                # operation `@. (1:J) + 1`. Note that we need the `nothing` because
                # `:macrocall` expects a LineNumberNode, but we can't provide it.
                Expr(:macrocall, Symbol("@__dot__"), nothing, new_expr)
            else
                # ok to just replace.
                new_expr
            end
        else
            error("@plate: unsupported expression $(expr) with head $(expr.head)")
        end
    else
        # e.g. literals
        expr
    end
end
vectorise_plate_stmt(lnn::LineNumberNode, idx, iters, ax, cond, ls, ps) = lnn
function vectorise_plate_stmt(
    stmt::Sample,
    loop_idx,
    loop_iters,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    new_lhs = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters)
    rhs_expr = _wren_expr_to_expr(stmt.rhs)
    new_rhs = Expr(
        :call,
        Wren._product,
        Expr(:comprehension, Expr(:generator, rhs_expr, Expr(:(=), loop_idx, loop_iters))),
    )
    return emit_model_expr(
        Sample(new_lhs, new_rhs),
        accs,
        conditioned,
        latent_source,
        parameter_space,
    )
end
function vectorise_plate_stmt(
    stmt::Observe,
    loop_idx,
    loop_iters,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    new_lhs = replace_loop_idx_with_iters(stmt.lhs, loop_idx, loop_iters)
    rhs_expr = _wren_expr_to_expr(stmt.rhs)
    new_rhs = Expr(
        :call,
        Wren._product,
        Expr(:comprehension, Expr(:generator, rhs_expr, Expr(:(=), loop_idx, loop_iters))),
    )
    return emit_model_expr(
        Observe(new_lhs, new_rhs),
        accs,
        conditioned,
        latent_source,
        parameter_space,
    )
end

function emit_model_expr(stmt::Observe, accs, conditioned, latent_source, parameter_space)
    @gensym rhs val vn
    rhs_expr = emit_model_expr(stmt.rhs, accs, conditioned, latent_source, parameter_space)
    vn_expr = remove_escapes(varname(stmt.lhs, false))
    template = get_top_level_symbol(stmt.lhs)
    return quote
        $val = $(stmt.lhs)
        $rhs = $rhs_expr
        $vn = $vn_expr
        $accs = $(accumulate_observe!!)($accs, $val, $vn, $rhs, $template)
    end
end

function emit_model_expr(
    stmt::ReturnStatement,
    accs,
    conditioned,
    latent_source,
    parameter_space,
)
    body = Expr(:block)
    @gensym retval
    if stmt.expr !== nothing
        return_expr =
            emit_model_expr(stmt.expr, accs, conditioned, latent_source, parameter_space)
        push!(body.args, :($retval = $return_expr))
    else
        push!(body.args, :($retval = nothing))
    end
    push!(body.args, :(return ($retval, $accs)))
    return body
end

function emit_statements(stmts, accs, conditioned, latent_source, parameter_space)
    body = Expr(:block)
    for stmt in stmts
        push!(
            body.args,
            emit_model_expr(stmt, accs, conditioned, latent_source, parameter_space),
        )
    end
    return body
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
