# A crude, syntax-level, dataflow analysis for Sample statements in Wren model bodies.

get_used_symbols(::Any) = Set{Symbol}() # literals etc
get_used_symbols(::QuoteNode) = Set{Symbol}()
get_used_symbols(symbol::Symbol) = Set([symbol])
function get_used_symbols(expr::Expr)
    return union(get_used_symbols.(expr.args)...)
end

parse_variables!(::Any, ::Set{Symbol}, ::Set{Symbol}) = nothing # e.g. literal
parse_variables!(::T, ::Set{Symbol}, ::Set{Symbol}) where {T<:AbstractWrenExpr} =
    _issue_error("unhandled node of type $(T) in dataflow analysis")
parse_variables!(::LineNumberNode, ::Set{Symbol}, ::Set{Symbol}) = nothing
function parse_variables!(sym::Symbol, lhs_syms::Set{Symbol}, required_syms::Set{Symbol})
    if sym in lhs_syms
        push!(required_syms, sym)
    end
    return nothing
end
function parse_variables!(stmt::Sample, lhs_syms::Set{Symbol}, required_syms::Set{Symbol})
    push!(lhs_syms, stmt.lhs_top_level_symbol)
    used = get_used_symbols(stmt.rhs)
    union!(required_syms, intersect(lhs_syms, used))
    return nothing
end
function parse_variables!(stmt::Observe, lhs_syms::Set{Symbol}, required_syms::Set{Symbol})
    used = get_used_symbols(stmt.rhs)
    union!(required_syms, intersect(lhs_syms, used))
    return nothing
end
function parse_variables!(stmt::IID, lhs_syms::Set{Symbol}, required_syms::Set{Symbol})
    parse_variables!(stmt.tilde, lhs_syms, required_syms)
    parse_variables!(stmt.size, lhs_syms, required_syms)
    return nothing
end
function parse_variables!(stmt::Batch, lhs_syms::Set{Symbol}, required_syms::Set{Symbol})
    # TODO(penelopeysm): Inside the for loop, we know that the loop index is a new symbol,
    # so uses of it inside the loop body should not be considered as reusing a top-level
    # symbol. However, the chance that the loop index shadows a top-level symbol is pretty
    # low, so we can ignore this for now.
    parse_variables!(stmt.loop_iters, lhs_syms, required_syms)
    parse_variables!(stmt.size, lhs_syms, required_syms)
    parse_variables!(stmt.body, lhs_syms, required_syms)
    return nothing
end
function parse_variables!(
    stmt::AddLogLikelihood,
    lhs_syms::Set{Symbol},
    required_syms::Set{Symbol},
)
    parse_variables!(stmt.expr, lhs_syms, required_syms)
    return nothing
end
function parse_variables!(
    stmt::ReturnStatement,
    lhs_syms::Set{Symbol},
    required_syms::Set{Symbol},
)
    parse_variables!(stmt.expr, lhs_syms, required_syms)
    return nothing
end
function parse_variables!(
    stmt::OtherStatement,
    lhs_syms::Set{Symbol},
    required_syms::Set{Symbol},
)
    parse_variables!(stmt.expr, lhs_syms, required_syms)
    return nothing
end
function parse_variables!(expr::WrenExpr, lhs_syms::Set{Symbol}, required_syms::Set{Symbol})
    # TODO(penelopeysm): Handle for/while control flow VERY carefully! See description in
    # PR.
    if expr.head == :for || expr.head == :while
        _issue_error("uhoh")
    end

    # TODO(penelopeysm): This is very generic and thus suboptimal. For example, `x = 2` will
    # reassign to 2 and thus `x` should not be considered as reused. We could definitely do
    # better here. However, the current implementation is still technically correct, just a
    # bit conservative - so we can leave the optimisations to the future.
    for arg in expr.args
        parse_variables!(arg, lhs_syms, required_syms)
    end
    return nothing
end

function run_dataflow_analysis(stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}})
    # All top-level symbols on the LHS of Sample nodes
    lhs_syms = Set{Symbol}()
    # The subset of the above that were seen again in a later expression, which we thus need
    # the values of
    required_syms = Set{Symbol}()

    # Populate the `lhs_syms` and `required_syms` sets.
    for stmt in stmts
        parse_variables!(stmt, lhs_syms, required_syms)
    end

    if WREN_COMPILATION_DEBUG[]
        @info "[dataflow.jl] lhs_syms = $(lhs_syms)"
        @info "[dataflow.jl] required_syms = $(required_syms)"
    end

    # TODO: change this (obvs)
    return stmts
end
