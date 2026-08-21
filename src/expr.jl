module Expressions

abstract type AbstractWrenExpr end

# TODO: type bounds on L and R probably Union{Expr,Symbol}
struct Sample{L,R} <: AbstractWrenExpr
    lhs::L
    rhs::R
end

struct Observe{L,R} <: AbstractWrenExpr
    lhs::L
    rhs::R
end

struct ReturnStatement{E} <: AbstractWrenExpr
    expr::E
end

struct OtherStatement{E} <: AbstractWrenExpr
    expr::E
end

struct Plate{I,S,B} <: AbstractWrenExpr
    loop_idx::Symbol
    loop_iters::I
    size::S
    body::B
end

struct WrenExpr{A} <: AbstractWrenExpr
    head::Symbol
    args::A
end

function to_wren_exprs(expr, mod::Module)
    dest = Union{AbstractWrenExpr,LineNumberNode}[]
    if Meta.isexpr(expr, :block)
        for stmt in expr.args
            to_wren_exprs!(dest, stmt, mod)
        end
    else
        to_wren_exprs!(dest, expr, mod)
    end
    return dest
end

function to_wren_exprs!(dest, lnn::LineNumberNode, ::Module)
    push!(dest, lnn)
    return dest
end

function to_wren_exprs!(dest, expr, mod::Module)
    lowered = lower_wren_expr(expr, mod, true)
    push!(dest, lowered isa AbstractWrenExpr ? lowered : OtherStatement(lowered))
    return dest
end

lower_wren_expr(expr, ::Module, ::Bool=false) = expr

function lower_wren_expr(expr::Expr, mod::Module, statement_position::Bool=false)
    if expr.head == :macrocall && expr.args[1] == Symbol("@observe")
        statement_position || error("@observe must be used as a statement")
        observed_expr = last(expr.args)
        if !(
            observed_expr isa Expr &&
            observed_expr.head == :call &&
            observed_expr.args[1] == :~
        )
            error("@observe: expected an expression of the form `value ~ distribution`")
        end
        length(observed_expr.args) == 3 || error("@observe: expected ~ to have 2 arguments")
        lhs, rhs = observed_expr.args[2:3]
        return Observe(lhs, lower_wren_expr(rhs, mod))
    end

    if expr.head == :macrocall && expr.args[1] == Symbol("@plate")
        statement_position || error("@plate must be used as a statement")
        plate_args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
        size_expr = nothing
        for_expr = nothing
        for arg in plate_args
            if arg isa Expr && arg.head == :(=) && arg.args[1] == :size
                size_expr = arg.args[2]
            elseif arg isa Expr && arg.head == :for
                for_expr = arg
            else
                error("@plate: unexpected argument `$arg`")
            end
        end
        for_expr !== nothing || error("@plate: expected a for loop")
        loop_spec = for_expr.args[1]
        loop_spec isa Expr && loop_spec.head == :(=) ||
            error("@plate: expected `for VAR in ITER`")
        loop_idx = loop_spec.args[1]
        loop_idx isa Symbol || error("@plate: loop variable must be a single identifier")
        loop_iters = loop_spec.args[2]
        body = lower_wren_expr(for_expr.args[2], mod, true)
        return Plate(loop_idx, loop_iters, size_expr, body)
    end

    # Preserve external macros for Julia to expand after Wren has lowered any
    # probabilistic statements that are visible in their arguments. In particular,
    # expanding a control-flow macro here may hide its body inside generated closures.
    # Macros whose expansions introduce new probabilistic statements are unsupported.
    if expr.head == :macrocall
        args = map(enumerate(expr.args)) do (index, arg)
            index <= 2 ? arg : lower_wren_expr(arg, mod, statement_position)
        end
        return WrenExpr(:macrocall, args)
    end

    if expr.head == :call && expr.args[1] == :~
        length(expr.args) == 3 || error("@wren: expected ~ to have 2 arguments")
        lhs, rhs = expr.args[2:3]
        return Sample(lhs, lower_wren_expr(rhs, mod))
    end

    if expr.head == :return
        value = only(expr.args)
        return ReturnStatement(value === nothing ? nothing : lower_wren_expr(value, mod))
    end

    if expr.head in (:quote, :function, :(->), :macro, :module, :baremodule)
        return OtherStatement(expr)
    end

    args = map(enumerate(expr.args)) do (index, arg)
        child_is_statement =
            expr.head in (:block, :toplevel) ||
            expr.head in (:if, :elseif) && index >= 2 ||
            expr.head in (:while, :for) && index == 2 ||
            expr.head == :let && index == length(expr.args)
        lower_wren_expr(arg, mod, child_is_statement)
    end
    return WrenExpr(expr.head, args)
end

_wren_expr_to_expr(expr) = expr
function _wren_expr_to_expr(stmt::WrenExpr)
    return Expr(stmt.head, map(_wren_expr_to_expr, stmt.args)...)
end
function _wren_expr_to_expr(stmt::Plate)
    body = _wren_expr_to_expr(stmt.body)
    for_expr = Expr(:for, Expr(:(=), stmt.loop_idx, stmt.loop_iters), body)
    args = Any[Symbol("@plate"), nothing]
    if stmt.size !== nothing
        push!(args, Expr(:(=), :size, stmt.size))
    end
    push!(args, for_expr)
    return Expr(:macrocall, args...)
end
function _wren_expr_to_expr(stmt::Sample)
    return Expr(:call, :~, stmt.lhs, _wren_expr_to_expr(stmt.rhs))
end
function _wren_expr_to_expr(stmt::Observe)
    tilde = Expr(:call, :~, stmt.lhs, _wren_expr_to_expr(stmt.rhs))
    return Expr(:macrocall, Symbol("@observe"), nothing, tilde)
end
function _wren_expr_to_expr(stmt::ReturnStatement)
    value = stmt.expr === nothing ? nothing : _wren_expr_to_expr(stmt.expr)
    return Expr(:return, value)
end
_wren_expr_to_expr(stmt::OtherStatement) = stmt.expr

function _show_wren_expr(io::IO, stmt::AbstractWrenExpr, indent::Int=0)
    expr = deepcopy(_wren_expr_to_expr(stmt))
    expr isa Expr && Base.remove_linenums!(expr)
    return Base.show_unquoted(io, expr, indent, 0)
end

function _show_wren_exprs(io::IO, stmts, indent::Int=0)
    first = true
    for stmt in stmts
        stmt isa AbstractWrenExpr || continue
        first || println(io)
        print(io, ' '^indent)
        _show_wren_expr(io, stmt, indent)
        first = false
    end
    return nothing
end

Base.show(io::IO, stmt::AbstractWrenExpr) = _show_wren_expr(io, stmt)
function Base.show(io::IO, ::MIME"text/plain", stmt::AbstractWrenExpr)
    return _show_wren_expr(io, stmt)
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    stmts::AbstractVector{<:Union{AbstractWrenExpr,LineNumberNode}},
)
    println(io, "Wren expressions:")
    return _show_wren_exprs(io, stmts, 2)
end

end # module Expressions
