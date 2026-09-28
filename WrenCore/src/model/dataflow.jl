# A crude, syntax-level, dataflow analysis for Sample statements in Wren model bodies.

using Accessors: Accessors

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
    # TODO(penelopeysm): This is very generic and thus suboptimal. For example:
    #
    #  - `x = 2` will reassign to 2 and thus `x` should not be considered as reused.
    #
    #  - if/elseif/else branches are mutually exclusive, so uses of a symbol in the if
    #    branch should not be considered as reusing a symbol in the else branch. e.g.
    #
    #        @wren function g()
    #            if x
    #                a ~ Normal()
    #            else
    #                b = a  # Should not count as a use of `a` in the if branch.
    #            end
    #        end
    #
    # We could definitely do better here. However, the current implementation is still
    # technically correct, just a bit conservative - so we can leave the optimisations to
    # the future.
    for arg in expr.args
        parse_variables!(arg, lhs_syms, required_syms)
    end

    # For and while loop bodies have to be traversed twice to make sure we pick up edges
    # between different iterations of the loop. See e.g. discussion in
    # https://github.com/JuliaBayes/Wren.jl/pull/124.
    if expr.head === :for
        # For a `for` loop, the loop iters (`expr.args[1]`) are only evaluated once at the
        # beginning of the loop, so we can skip it.
        for arg in expr.args[2:end]
            parse_variables!(arg, lhs_syms, required_syms)
        end
    elseif expr.head === :while
        # For a `while` loop the condition is evaluated on every iteration so we need to
        # include it.
        for arg in expr.args
            parse_variables!(arg, lhs_syms, required_syms)
        end
    end
    return nothing
end

reconstruct_statement(@nospecialize(x::Any), ::Set{Symbol}) = x
function reconstruct_statement(stmt::Sample, required_syms::Set{Symbol})
    return Accessors.@set stmt.lhs_is_reused = stmt.lhs_top_level_symbol in required_syms
end
reconstruct_statement(stmt::Observe, required_syms::Set{Symbol}) = stmt
function reconstruct_statement(stmt::IID, required_syms::Set{Symbol})
    return Accessors.@set stmt.tilde = reconstruct_statement(stmt.tilde, required_syms)
end
function reconstruct_statement(expr::WrenExpr, required_syms::Set{Symbol})
    new_args = map(arg -> reconstruct_statement(arg, required_syms), expr.args)
    return WrenExpr(expr.head, new_args)
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

    new_stmts = map(stmt -> reconstruct_statement(stmt, required_syms), stmts)
    return Vector{Union{AbstractWrenExpr,LineNumberNode}}(new_stmts)
end
