# Only infer aliases when each completed evaluation reaches one straight-line branch.
_alias_straight_expr(::Any) = true
_alias_straight_expr(::AbstractWrenExpr) = false
_alias_straight_expr(stmt::OtherStatement) = _alias_straight_expr(stmt.expr)
function _alias_straight_expr(expr::Union{Expr,WrenExpr})
    return !(
        expr.head in (
            :if,
            :elseif,
            :for,
            :while,
            :try,
            :macrocall,
            :return,
            :break,
            :continue,
            :goto,
            :gotoifnot,
            :symbolicgoto,
            :symboliclabel,
            :function,
            :(->),
            :quote,
        )
    ) && all(_alias_straight_expr, expr.args)
end

_alias_sample_counts!(counts, ::Any) = true
# Unknown IR nodes may hide sampling or control flow, so they disable inference.
_alias_sample_counts!(counts, ::AbstractWrenExpr) = false
function _alias_sample_counts!(counts, stmt::Sample)
    sym = stmt.lhs_top_level_symbol
    counts[sym] = get(counts, sym, 0) + 1
    return _alias_sample_counts!(counts, stmt.lhs) &&
           _alias_sample_counts!(counts, stmt.rhs)
end
function _alias_sample_counts!(counts, expr::Union{Expr,WrenExpr})
    expr.head in (:macrocall, :return, :goto, :gotoifnot, :symbolicgoto, :symboliclabel) &&
        return false
    return all(arg -> _alias_sample_counts!(counts, arg), expr.args)
end
# These wrappers contain child expressions and inert names.
_alias_sample_counts!(
    counts,
    stmt::Union{OtherStatement,Observe,IID,Batch,AddLogLikelihood},
) = all(
    name -> _alias_sample_counts!(counts, getfield(stmt, name)),
    fieldnames(typeof(stmt)),
)

function _alias_branch_symbols!(names, stmt::Sample, conditioned)
    stmt.lhs isa Symbol && _alias_straight_expr(stmt.rhs) || return false
    sym = stmt.lhs
    if !haskey(conditioned, VarName{sym}())
        sym in names && return false
        push!(names, sym)
    end
    return true
end
function _alias_branch_symbols!(names, expr::WrenExpr, conditioned)
    return if expr.head === :block
        all(arg -> _alias_branch_symbols!(names, arg, conditioned), expr.args)
    else
        _alias_straight_expr(expr)
    end
end
_alias_branch_symbols!(names, stmt::Observe, conditioned) =
    _alias_straight_expr(stmt.lhs) && _alias_straight_expr(stmt.rhs)
_alias_branch_symbols!(names, stmt::AddLogLikelihood, conditioned) =
    _alias_straight_expr(stmt.expr)
_alias_branch_symbols!(names, stmt, conditioned) = _alias_straight_expr(stmt)

function _infer_aliases(model::Model)
    stmts = filter(stmt -> !(stmt isa LineNumberNode), model.stmts)
    counts = Dict{Symbol,Int}()
    for (i, stmt) in enumerate(stmts)
        # A final top-level return cannot bypass an earlier candidate.
        expr = i == length(stmts) && stmt isa ReturnStatement ? stmt.expr : stmt
        _alias_sample_counts!(counts, expr) || return ()
    end
    groups = Tuple{VarName,VarName}[]
    for stmt in stmts
        stmt isa WrenExpr && stmt.head === :if && length(stmt.args) == 3 || continue
        _alias_straight_expr(stmt.args[1]) || continue
        left, right = Set{Symbol}(), Set{Symbol}()
        _alias_branch_symbols!(left, stmt.args[2], model.conditioned) || continue
        _alias_branch_symbols!(right, stmt.args[3], model.conditioned) || continue
        # Common names keep their coordinates even when the branches change their order.
        left_only, right_only = setdiff(left, right), setdiff(right, left)
        length(left_only) == length(right_only) == 1 || continue
        a, b = only(left_only), only(right_only)
        # Root counts also exclude indexed writes to either proposed alias elsewhere.
        counts[a] == counts[b] == 1 || continue
        push!(groups, (VarName{a}(), VarName{b}()))
    end
    return Tuple(groups)
end
