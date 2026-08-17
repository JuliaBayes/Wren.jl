_model_expr_to_expr(stmt::Sample) = Expr(:call, :~, stmt.lhs, stmt.rhs)
function _model_expr_to_expr(stmt::Observe)
    tilde = Expr(:call, :~, stmt.lhs, stmt.rhs)
    return Expr(:macrocall, Symbol("@observe"), nothing, tilde)
end
_model_expr_to_expr(stmt::Deterministic) = Expr(:(=), stmt.lhs, stmt.rhs)
_model_expr_to_expr(stmt::ReturnStatement) = Expr(:return, stmt.expr)
_model_expr_to_expr(stmt::OtherStatement) = stmt.expr

function _model_exprs_to_block(stmts)
    return Expr(:block, map(_model_expr_to_expr, filter(x -> x isa ModelExpr, stmts))...)
end

function _model_if_to_expr(stmt::IfStatement, head::Symbol=:if)
    args = Any[stmt.condition, _model_exprs_to_block(stmt.then_branch)]
    if stmt.else_branch isa IfStatement
        push!(args, _model_if_to_expr(stmt.else_branch, :elseif))
    elseif stmt.else_branch !== nothing
        push!(args, _model_exprs_to_block(stmt.else_branch))
    end
    return Expr(head, args...)
end

_model_expr_to_expr(stmt::IfStatement) = _model_if_to_expr(stmt)
function _model_expr_to_expr(stmt::WhileStatement)
    return Expr(:while, stmt.condition, _model_exprs_to_block(stmt.body))
end
function _model_expr_to_expr(stmt::ForStatement)
    return Expr(:for, stmt.iterator, _model_exprs_to_block(stmt.body))
end
function _model_expr_to_expr(stmt::LocalStatement)
    rhs = _model_expr_to_expr(stmt.rhs)
    return Expr(:local, Expr(:(=), stmt.lhs, rhs))
end

function _show_model_expr(io::IO, stmt::ModelExpr, indent::Int=0)
    expr = deepcopy(_model_expr_to_expr(stmt))
    expr isa Expr && Base.remove_linenums!(expr)
    return Base.show_unquoted(io, expr, indent, 0)
end

function _show_model_exprs(io::IO, stmts, indent::Int=0)
    first = true
    for stmt in stmts
        stmt isa ModelExpr || continue
        first || println(io)
        print(io, ' '^indent)
        _show_model_expr(io, stmt, indent)
        first = false
    end
    return nothing
end

Base.show(io::IO, stmt::ModelExpr) = _show_model_expr(io, stmt)
Base.show(io::IO, ::MIME"text/plain", stmt::ModelExpr) = _show_model_expr(io, stmt)

function Base.show(
    io::IO,
    ::MIME"text/plain",
    stmts::AbstractVector{<:Union{ModelExpr,LineNumberNode}},
)
    println(io, "Model expressions:")
    return _show_model_exprs(io, stmts, 2)
end

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    println(io, "Model")
    print(io, "  arguments: ")
    show(io, model.args)
    print(io, "\n  conditioned: ")
    show(io, model.conditioned)
    println(io, "\n  expressions:")
    return _show_model_exprs(io, model.stmts, 4)
end
