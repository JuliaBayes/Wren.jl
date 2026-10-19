# A crude, syntax-level, dataflow analysis for Sample statements in Wren model bodies.

using Accessors: Accessors

get_used_symbols(::Any) = Set{Symbol}() # literals etc
get_used_symbols(::QuoteNode) = Set{Symbol}()
get_used_symbols(symbol::Symbol) = Set([symbol])
function get_used_symbols(expr::Union{Expr,WrenExpr})
    # Skip names of keyword arguments and names of macros
    args_to_check = if expr.head === :kw || expr.head === :macrocall
        expr.args[2:end]
    else
        expr.args
    end
    return union(Set{Symbol}(), get_used_symbols.(args_to_check)...)
end
get_used_symbols(stmt::OtherStatement) = get_used_symbols(stmt.expr)
get_used_symbols(stmt::Union{AddLogLikelihood,ReturnStatement}) =
    get_used_symbols(stmt.expr)
get_used_symbols(stmt::Sample) = union(
    Set([stmt.lhs_top_level_symbol]),
    get_used_symbols(stmt.lhs),
    get_used_symbols(stmt.rhs),
)
get_used_symbols(stmt::Observe) =
    union(get_used_symbols(stmt.lhs), get_used_symbols(stmt.rhs))
get_used_symbols(stmt::IID) =
    union(get_used_symbols(stmt.tilde), get_used_symbols(stmt.size))
get_used_symbols(stmt::Batch) = union(
    Set([stmt.loop_idx]),
    get_used_symbols(stmt.loop_iters),
    get_used_symbols(stmt.size),
    get_used_symbols(stmt.body),
)
function get_used_symbols(::AbstractWrenExpr)
    _issue_error("unhandled node in get_used_symbols")
end

get_lhs_index_symbols(::Symbol) = Set{Symbol}()
function get_lhs_index_symbols(expr::Expr)
    return if Meta.isexpr(expr, :ref)
        union(get_lhs_index_symbols(expr.args[1]), get_used_symbols.(expr.args[2:end])...)
    elseif Meta.isexpr(expr, :.)
        get_lhs_index_symbols(expr.args[1])
    else
        get_used_symbols(expr)
    end
end

struct DataflowAnalysisState
    # All top-level symbols on the LHS of Sample nodes
    lhs_syms::Set{Symbol}
    # The subset of the above that were seen again in a later expression, which we thus need
    # the values of
    required_syms::Set{Symbol}
    # All model arguments
    args::Set{Symbol}
    # Arguments which have potentially been written to
    dirty_args::Set{Symbol}
    # Dirty arguments which have potentially been read from again
    args_to_copy::Set{Symbol}
end
function mark_lhs_symbol!(state::DataflowAnalysisState, sym::Symbol)
    # Record that we've seen `sym` on the LHS of a tilde
    push!(state.lhs_syms, sym)
    return nothing
end
function mark_symbol_used!(state::DataflowAnalysisState, syms::Set{Symbol})
    # Record that we've used `syms` in a later expression. In general, 'use' means
    # that the symbol may have been both written to and read from.
    union!(state.required_syms, intersect(state.lhs_syms, syms))
    args = intersect(state.args, syms)
    union!(state.dirty_args, args)
    union!(state.args_to_copy, args)
    return nothing
end
function mark_as_written_to!(state::DataflowAnalysisState, syms::Set{Symbol})
    # Record that we've written to `syms` in a later expression. This is a special case for
    # the lhs of Sample statements which we know are only written to, not read from.
    union!(state.dirty_args, intersect(state.args, syms))
    return nothing
end
function mark_as_read_from!(state::DataflowAnalysisState, syms::Set{Symbol})
    # Record that we've read from `syms` in a later expression. This is a special case for
    # the lhs of Observe statements which we know are only read from, not written to.
    union!(state.required_syms, intersect(state.lhs_syms, syms))
    union!(state.args_to_copy, intersect(state.dirty_args, syms))
    return nothing
end

parse_variables!(::Any, ::DataflowAnalysisState) = nothing # e.g. literal
parse_variables!(::T, ::DataflowAnalysisState) where {T<:AbstractWrenExpr} =
    _issue_error("unhandled node of type $(T) in dataflow analysis")
parse_variables!(::LineNumberNode, ::DataflowAnalysisState) = nothing
function parse_variables!(sym::Symbol, state::DataflowAnalysisState)
    mark_symbol_used!(state, Set([sym]))
    return nothing
end
function parse_variables!(expr::Expr, state::DataflowAnalysisState)
    # Raw Julia expressions, e.g. inside `OtherStatement`s or `@batch` loop iterators.
    used = get_used_symbols(expr)
    # TODO(penelopeysm): This is a very conservative analysis because it assumes that
    # any appearance of a symbol in an expression counts as BOTH a write and a read. This
    # can happen in, for example,
    #
    #  f(x, y) = x .= y; return x
    #
    # where `x` is written to and then subsequently read from. This is kind of pathological
    # but sadly we have to be safe in the presence of arbitrary Julia code.
    mark_symbol_used!(state, used)
    return nothing
end
function parse_variables!(stmt::Sample, state::DataflowAnalysisState)
    # In e.g. `x[i] ~ y`, `i` and `y` are considered uses, so we need to check both lhs and
    # rhs. However `x` itself is not considered used since it's only written to.
    used = union(get_lhs_index_symbols(stmt.lhs), get_used_symbols(stmt.rhs))
    mark_symbol_used!(state, used)
    # TODO(penelopeysm): Appearing on the LHS of a Sample statement doesn't necessarily mean
    # that the symbol is written to, since it might have been conditioned on. So this is
    # overly conservative. See https://github.com/JuliaBayes/Wren.jl/issues/139.
    #
    # Note: in the case where it's conditioned on, the symbol isn't read from because Wren
    # will read from the `model.conditioned` VNT instead. This might change if #139 is
    # implemented.
    mark_as_written_to!(state, Set([stmt.lhs_top_level_symbol]))
    # Record the top-level LHS symbol itself.
    mark_lhs_symbol!(state, stmt.lhs_top_level_symbol)
    return nothing
end
function parse_variables!(stmt::Observe, state::DataflowAnalysisState)
    lhs_used = get_used_symbols(stmt.lhs)
    rhs_used = get_used_symbols(stmt.rhs)
    # LHS is only read from, but RHS can in general be either.
    mark_as_read_from!(state, lhs_used)
    mark_symbol_used!(state, rhs_used)
    return nothing
end
function parse_variables!(stmt::IID, state::DataflowAnalysisState)
    parse_variables!(stmt.tilde, state)
    parse_variables!(stmt.size, state)
    return nothing
end
function parse_variables!(stmt::Batch, state::DataflowAnalysisState)
    # TODO(penelopeysm): Inside the for loop, we know that the loop index is a new symbol,
    # so uses of it inside the loop body should not be considered as reusing a top-level
    # symbol. However, the chance that the loop index shadows a top-level symbol is pretty
    # low, so we can ignore this for now.
    parse_variables!(stmt.loop_iters, state)
    parse_variables!(stmt.size, state)
    parse_variables!(stmt.body, state)
    return nothing
end
function parse_variables!(stmt::AddLogLikelihood, state::DataflowAnalysisState)
    parse_variables!(stmt.expr, state)
    return nothing
end
function parse_variables!(stmt::ReturnStatement, state::DataflowAnalysisState)
    parse_variables!(stmt.expr, state)
    return nothing
end
function parse_variables!(stmt::OtherStatement, state::DataflowAnalysisState)
    parse_variables!(stmt.expr, state)
    return nothing
end
function parse_variables!(expr::WrenExpr, state::DataflowAnalysisState)
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
        parse_variables!(arg, state)
    end

    # For and while loop bodies have to be traversed twice to make sure we pick up edges
    # between different iterations of the loop. See e.g. discussion in
    # https://github.com/JuliaBayes/Wren.jl/pull/124.
    if expr.head === :for
        # For a `for` loop, the loop iters (`expr.args[1]`) are only evaluated once at the
        # beginning of the loop, so we can skip it.
        for arg in expr.args[2:end]
            parse_variables!(arg, state)
        end
    elseif expr.head === :while
        # For a `while` loop the condition is evaluated on every iteration so we need to
        # include it.
        for arg in expr.args
            parse_variables!(arg, state)
        end
    end
    return nothing
end

# Latent values may be views of the caller's storage (#156). If the model body could write
# into an existing object, every latent is copied; otherwise only the returned ones are.

_isexpr(x, heads...) = x isa Union{Expr,WrenExpr} && x.head in heads

# Symbols whose values the value of `x` may contain.
value_symbols(x) = get_used_symbols(x)
value_symbols(stmt::Sample) = Set([stmt.lhs_top_level_symbol])
value_symbols(stmt::IID) = value_symbols(stmt.tilde)
value_symbols(::Union{Observe,AddLogLikelihood}) = Set{Symbol}() # evaluate to the accs
value_symbols(stmt::Batch) = value_symbols(stmt.body)
value_symbols(stmt::OtherStatement) = value_symbols(stmt.expr)
function value_symbols(expr::Union{Expr,WrenExpr})
    expr.head in (:for, :while) && return Set{Symbol}()
    if expr.head === :block
        stmts = filter(a -> !(a isa LineNumberNode), expr.args)
        return isempty(stmts) ? Set{Symbol}() : value_symbols(last(stmts))
    end
    expr.head in (:if, :elseif) &&
        return union(Set{Symbol}(), value_symbols.(expr.args[2:end])...)
    return get_used_symbols(expr)
end

collect_bindings!(::Any, s) = nothing
function collect_bindings!(stmt::Sample, s)
    push!(s.sampled, stmt.lhs_top_level_symbol)
    push!(stmt.lhs isa Symbol ? s.whole : s.indexed, stmt.lhs_top_level_symbol)
    collect_bindings!(stmt.rhs, s)
end
collect_bindings!(stmt::IID, s) = collect_bindings!(stmt.tilde, s)
function collect_bindings!(stmt::Batch, s)
    push!(s.bound, stmt.loop_idx)
    collect_bindings!(stmt.body, s)
end
collect_bindings!(stmt::Union{AddLogLikelihood,OtherStatement}, s) =
    collect_bindings!(stmt.expr, s)
function collect_bindings!(stmt::ReturnStatement, s)
    union!(s.returned, value_symbols(stmt.expr))
    collect_bindings!(stmt.expr, s)
end
function collect_bindings!(expr::Union{Expr,WrenExpr}, s)
    if expr.head === :return
        union!(s.returned, get_used_symbols(expr))
    elseif expr.head !== :kw && endswith(string(expr.head), "=")
        union!(s.bound, get_used_symbols(expr.args[1]))
    end
    foreach(arg -> collect_bindings!(arg, s), expr.args)
end

const SAFE_MACROS = Set(Symbol.(["@inbounds", "@fastmath", "@simd", "@view", "@views"]))
const SAFE_HEADS = Set([
    :block,
    :if,
    :elseif,
    :&&,
    :||,
    :comparison,
    :tuple,
    :vect,
    :vcat,
    :hcat,
    :row,
    :ref,
    :curly,
    :(::),
    :string,
    :...,
    :parameters,
    :kw,
    :for,
    :while,
    :let,
    :generator,
    :filter,
    :comprehension,
    :break,
    :continue,
    :return,
])

# Functions bound in the model body or passed in as arguments could be anything.
function may_write_args(f, bound)
    f isa Symbol && return f in bound || (endswith(string(f), "!") && f ∉ (:!, :.!))
    _isexpr(f, :.) && f.args[2] isa QuoteNode && return may_write_args(f.args[2].value, ())
    return true
end

may_write(::Any, bound) = false
may_write(::T, bound) where {T<:AbstractWrenExpr} =
    _issue_error("unhandled node of type $(T) in may_write")
may_write(stmt::Union{Sample,Observe}, bound) =
    may_write(stmt.lhs, bound) || may_write(stmt.rhs, bound)
may_write(stmt::IID, bound) = may_write(stmt.tilde, bound) || may_write(stmt.size, bound)
may_write(stmt::Batch, bound) =
    any(x -> may_write(x, bound), (stmt.loop_iters, stmt.size, stmt.body))
may_write(stmt::Union{AddLogLikelihood,ReturnStatement,OtherStatement}, bound) =
    may_write(stmt.expr, bound)
function may_write(expr::Union{Expr,WrenExpr}, bound)
    head, args = expr.head, expr.args
    writes(xs) = any(x -> may_write(x, bound), xs)
    return if head === :(=)
        lhs = args[1]
        binds_names =
            lhs isa Symbol || (_isexpr(lhs, :tuple) && all(a -> a isa Symbol, lhs.args))
        !binds_names || writes(args)
    elseif head !== :kw && endswith(string(head), "=")
        # `x += v` rebinds `x`, but `x[i] += v` and `x .= v` write into it.
        !(args[1] isa Symbol) || startswith(string(head), ".") || writes(args)
    elseif head === :call
        may_write_args(args[1], bound) || writes(args)
    elseif head === :. && args[2] isa QuoteNode
        may_write(args[1], bound)
    elseif head === :. && _isexpr(args[2], :tuple)
        may_write_args(args[1], bound) || writes(args[2].args)
    elseif head === :macrocall
        !(args[1] in SAFE_MACROS) || writes(args[3:end])
    else
        !(head in SAFE_HEADS) || writes(args)
    end
end

# The symbols of the Samples whose values must be copied before the model sees them.
function get_mutated_syms(
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    args::Set{Symbol},
)
    s = (;
        bound=Set{Symbol}(),
        sampled=Set{Symbol}(),
        whole=Set{Symbol}(),
        indexed=Set{Symbol}(),
        returned=Set{Symbol}(),
    )
    foreach(stmt -> collect_bindings!(stmt, s), stmts)
    body = filter(stmt -> !(stmt isa LineNumberNode), stmts)
    isempty(body) || union!(s.returned, value_symbols(last(body)))

    copy_all =
        any(stmt -> may_write(stmt, union(s.bound, args)), stmts) ||
        # `x[i] ~ d` writes into `x`, which may be (or contain) a latent.
        any(x -> x ∉ args || x in s.whole, s.indexed) ||
        # A returned value computed in the body may contain a latent.
        !isempty(setdiff(intersect(s.returned, s.bound), s.sampled))
    mutated_syms = copy_all ? s.sampled : intersect(s.returned, s.sampled)

    if WREN_COMPILATION_DEBUG[]
        @info "[dataflow.jl] mutated_syms = $(mutated_syms)"
    end
    return mutated_syms
end

reconstruct_statement(@nospecialize(x::Any), ::Set{Symbol}, ::Set{Symbol}) = x
reconstruct_statement(::T, ::Set{Symbol}, ::Set{Symbol}) where {T<:AbstractWrenExpr} =
    _issue_error("unhandled node of type $(T) in reconstruct_statement")
function reconstruct_statement(
    stmt::Sample,
    required_syms::Set{Symbol},
    mutated_syms::Set{Symbol},
)
    stmt = Accessors.@set stmt.lhs_is_reused = stmt.lhs_top_level_symbol in required_syms
    return Accessors.@set stmt.lhs_is_mutated = stmt.lhs_top_level_symbol in mutated_syms
end
reconstruct_statement(stmt::Observe, ::Set{Symbol}, ::Set{Symbol}) = stmt
# The contents of an OtherStatement are not lowered, so they can't contain Samples.
reconstruct_statement(stmt::OtherStatement, ::Set{Symbol}, ::Set{Symbol}) = stmt
function reconstruct_statement(
    stmt::IID,
    required_syms::Set{Symbol},
    mutated_syms::Set{Symbol},
)
    return Accessors.@set stmt.tilde =
        reconstruct_statement(stmt.tilde, required_syms, mutated_syms)
end
function reconstruct_statement(
    stmt::Batch,
    required_syms::Set{Symbol},
    mutated_syms::Set{Symbol},
)
    return Accessors.@set stmt.body =
        reconstruct_statement(stmt.body, required_syms, mutated_syms)
end
function reconstruct_statement(
    stmt::Union{AddLogLikelihood,ReturnStatement},
    required_syms::Set{Symbol},
    mutated_syms::Set{Symbol},
)
    return Accessors.@set stmt.expr =
        reconstruct_statement(stmt.expr, required_syms, mutated_syms)
end
function reconstruct_statement(
    expr::WrenExpr,
    required_syms::Set{Symbol},
    mutated_syms::Set{Symbol},
)
    new_args =
        map(arg -> reconstruct_statement(arg, required_syms, mutated_syms), expr.args)
    return WrenExpr(expr.head, new_args)
end

function get_required_syms_and_args(
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    args::Set{Symbol},
)
    state = DataflowAnalysisState(
        Set{Symbol}(),
        Set{Symbol}(),
        args,
        Set{Symbol}(),
        Set{Symbol}(),
    )

    # Populate the sets in `state`.
    for stmt in stmts
        parse_variables!(stmt, state)
    end

    if WREN_COMPILATION_DEBUG[]
        @info "[dataflow.jl] lhs_syms = $(state.lhs_syms)"
        @info "[dataflow.jl] required_syms = $(state.required_syms)"
        @info "[dataflow.jl] args_to_copy = $(state.args_to_copy)"
    end

    return state.required_syms, state.args_to_copy
end

function run_dataflow_analysis(
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    args::Set{Symbol},
)
    required_syms, args_to_copy = get_required_syms_and_args(stmts, args)
    mutated_syms = get_mutated_syms(stmts, args)
    new_stmts = map(stmt -> reconstruct_statement(stmt, required_syms, mutated_syms), stmts)
    return Vector{Union{AbstractWrenExpr,LineNumberNode}}(new_stmts), args_to_copy
end
