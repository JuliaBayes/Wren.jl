_model_expr_to_expr(expr) = expr
function _model_expr_to_expr(stmt::JuliaExpr)
    return Expr(stmt.head, map(_model_expr_to_expr, stmt.args)...)
end
function _model_expr_to_expr(stmt::Sample)
    return Expr(:call, :~, stmt.lhs, _model_expr_to_expr(stmt.rhs))
end
function _model_expr_to_expr(stmt::Observe)
    tilde = Expr(:call, :~, stmt.lhs, _model_expr_to_expr(stmt.rhs))
    return Expr(:macrocall, Symbol("@observe"), nothing, tilde)
end
function _model_expr_to_expr(stmt::ReturnStatement)
    value = stmt.expr === nothing ? nothing : _model_expr_to_expr(stmt.expr)
    return Expr(:return, value)
end
_model_expr_to_expr(stmt::OtherStatement) = stmt.expr

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
