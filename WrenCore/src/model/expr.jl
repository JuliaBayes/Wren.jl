module Expressions

"""
    AbstractWrenExpr

Wren's probabilistic intermediate representation. This is for the most part a copy of
Julia's `Expr` type, but with a few additional node types for probabilistic statements,
namely [`Sample`](@ref), [`Observe`](@ref), [`IID`](@ref), and [`Batch`](@ref).

The [`ReturnStatement`](@ref) type is used as a convenient wrapper around `Expr(:return,
...)` to allow for dispatch, but has the same meaning.

TODO: I think we should create `WrenExpr` types for every Julia expression type that we
handle, so that we can dispatch on them, but also so that users can create models more
easily without dealing with Julia's Expr internals.
"""
abstract type AbstractWrenExpr end

"""
    Sample

Representation of `lhs ~ rhs`.
"""
struct Sample{L,R} <: AbstractWrenExpr
    name::Symbol
    lhs::L
    rhs::R
end

"""
    Observe

Representation of `@observe lhs ~ rhs`. Note that `Sample` statements that are conditioned
on do become observations, but do not become `Observe` nodes in the AST. This is because the
conditioning status must be determined at runtime (consider e.g. `y[i] ~ Normal()` with a
partial conditioning array -- thus the status depends on the runtime value of `i`).
"""
struct Observe{L,R} <: AbstractWrenExpr
    name::Symbol
    lhs::L
    rhs::R
end

"""
    IID

Representation of `@iid lhs ~ rhs` (wrapping a [`Sample`](@ref)), or of
`@iid @observe lhs ~ rhs` / `@observe @iid lhs ~ rhs` (wrapping an [`Observe`](@ref)).

The wrapped tilde-statement is stored as written; the meaning `lhs ~ iid(rhs, size(lhs))`
is only given to it at the second stage of lowering, in `emit_model_expr`.
"""
struct IID{T<:Union{Sample,Observe},S} <: AbstractWrenExpr
    tilde::T
    size::S
end

"""
    Batch

Representation of

```julia
@batch [name=NAME] [size=SIZE] for LOOP_IDX in LOOP_ITERS
    BODY
end
```

`name` must be a single Symbol. If not specified, a gensym'd name is used.
If `size` is not specified, it will be `nothing`.
"""
struct Batch{I,S,B} <: AbstractWrenExpr
    name::Symbol
    loop_idx::Symbol
    loop_iters::I
    size::S
    body::B
end

"""
    AddLogLikelihood

Representation of `@addloglikelihood [name=NAME] EXPR`. Manually adds the value of `EXPR`
to the log-likelihood accumulator.

`name` must be a single Symbol. If not specified, a gensym'd name is used. The name is
intended for use with pointwise log-likelihoods.
"""
struct AddLogLikelihood{E} <: AbstractWrenExpr
    name::Symbol
    expr::E
end

"""
    ReturnStatement

Wrapper around `Expr(:return, ...)` to allow for dispatch.
"""
struct ReturnStatement{E} <: AbstractWrenExpr
    expr::E
end

"""
    OtherStatement

Any other statement without special probabilistic meaning. (Note, by statement, we mean a
top-level expression -- things like for loops, etc. become `WrenExpr` nodes. This might be
changed in the future.)
"""
struct OtherStatement{E} <: AbstractWrenExpr
    expr::E
end

"""
    WrenExpr

A direct representation of Julia's `Expr` type that subtypes `AbstractWrenExpr`.
"""
struct WrenExpr{A} <: AbstractWrenExpr
    head::Symbol
    args::A
end

"""
    to_wren_exprs(expr, mod::Module)

Convert a Julia expression into a vector of `AbstractWrenExpr`s.
"""
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

"""
    to_wren_exprs!(dest, expr, mod::Module)

Convert `expr` into `AbstractWrenExpr`s and push them to `dest`.
"""
function to_wren_exprs!(dest, lnn::LineNumberNode, ::Module)
    push!(dest, lnn)
    return dest
end
function to_wren_exprs!(dest, expr, mod::Module)
    lowered = lower_to_wren(expr, mod, true)
    push!(dest, lowered isa AbstractWrenExpr ? lowered : OtherStatement(lowered))
    return dest
end

"""
    lower_wren_expr(expr, mod::Module, statement_position::Bool=false)

The actual function that handles part 1 of lowering, i.e., transforming the user's model
into Wren's IR.
"""
lower_to_wren(expr, ::Module, ::Bool=false) = expr

_is_macrocall(expr, name::Symbol) = Meta.isexpr(expr, :macrocall) && expr.args[1] == name
_macro_args(expr::Expr) = filter(a -> !(a isa LineNumberNode), expr.args[2:end])

# Lower the `lhs ~ rhs` inside `@observe lhs ~ rhs` into an `Observe` node.
function _lower_observe(observed_expr, mod::Module)
    if !(
        observed_expr isa Expr &&
        observed_expr.head == :call &&
        observed_expr.args[1] == :~
    )
        error("@observe: expected an expression of the form `value ~ distribution`")
    end
    for arg in _macro_args(observed_expr)
        arg.args[1] == :size &&
            error(
                "@observe: invalid `size` argument, uses `@iid` as the innermost macro"
            )
    end
    length(observed_expr.args) == 3 || error("@observe: expected ~ to have 2 arguments")
    lhs, rhs = observed_expr.args[2:3]
    return Observe(Symbol(lhs), lhs, lower_to_wren(rhs, mod))
end

# Extract the optional `size=...` argument and the tilde-statement of `@iid ...`.
function _iid_args(expr::Expr)
    size_expr = nothing
    inner = nothing
    for arg in _macro_args(expr)
        if arg isa Expr && arg.head == :(=) && arg.args[1] == :size
            size_expr = arg.args[2]
        elseif inner === nothing
            inner = arg
        else
            error("@iid: unexpected argument `$arg`")
        end
    end
    inner === nothing &&
        error("@iid: expected a single expression of the form `value ~ distribution`")
    return size_expr, inner
end

function lower_to_wren(expr::Expr, mod::Module, statement_position::Bool=false)
    if expr.head == :macrocall && expr.args[1] == Symbol("@observe")
        statement_position || error("@observe must be used as a statement")
        observed_expr = last(expr.args)
        if _is_macrocall(observed_expr, Symbol("@iid"))
            # @observe @iid lhs ~ rhs
            size_expr, inner = _iid_args(observed_expr)
            return IID(_lower_observe(inner, mod), size_expr)
        end
        return _lower_observe(observed_expr, mod)
    end

    if expr.head == :macrocall && expr.args[1] == Symbol("@iid")
        statement_position || error("@iid must be used as a statement")
        size_expr, inner = _iid_args(expr)
        if _is_macrocall(inner, Symbol("@observe"))
            # @iid @observe lhs ~ rhs
            return IID(_lower_observe(last(inner.args), mod), size_expr)
        end
        # @iid lhs ~ rhs
        lowered = lower_to_wren(inner, mod, true)
        lowered isa Sample ||
            error("@iid: expected an expression of the form `value ~ distribution`")
        return IID(lowered, size_expr)
    end

    if expr.head == :macrocall && expr.args[1] == Symbol("@batch")
        statement_position || error("@batch must be used as a statement")
        plate_args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
        size_expr = nothing
        name_expr = nothing
        for_expr = nothing
        for arg in plate_args
            if arg isa Expr && arg.head == :(=) && arg.args[1] == :size
                size_expr = arg.args[2]
            elseif arg isa Expr && arg.head == :(=) && arg.args[1] == :name
                name_val = arg.args[2]
                name_val isa QuoteNode && name_val.value isa Symbol ||
                    error("@batch: `name` must be a single Symbol (e.g. `name=:obs`)")
                name_expr = name_val.value
            elseif arg isa Expr && arg.head == :for
                for_expr = arg
            else
                error("@batch: unexpected argument `$arg`")
            end
        end
        for_expr !== nothing || error("@batch: expected a for loop")
        loop_spec = for_expr.args[1]
        loop_spec isa Expr && loop_spec.head == :(=) ||
            error("@batch: expected `for VAR in ITER`")
        loop_idx = loop_spec.args[1]
        loop_idx isa Symbol || error("@batch: loop variable must be a single identifier")
        loop_iters = loop_spec.args[2]
        body = lower_to_wren(for_expr.args[2], mod, true)
        plate_name = name_expr === nothing ? gensym(:batch) : name_expr
        return Batch(plate_name, loop_idx, loop_iters, size_expr, body)
    end

    if expr.head == :macrocall && expr.args[1] == Symbol("@addloglikelihood")
        statement_position || error("@addloglikelihood must be used as a statement")
        addloglik_args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
        name_expr = nothing
        value_expr = nothing
        for arg in addloglik_args
            if arg isa Expr && arg.head == :(=) && arg.args[1] == :name
                name_val = arg.args[2]
                name_val isa QuoteNode && name_val.value isa Symbol || error(
                    "@addloglikelihood: `name` must be a single Symbol (e.g. `name=:penalty`)",
                )
                name_expr = name_val.value
            else
                value_expr !== nothing &&
                    error("@addloglikelihood: expected a single expression")
                value_expr = arg
            end
        end
        value_expr !== nothing || error("@addloglikelihood: expected an expression")
        addloglik_name = name_expr === nothing ? gensym(:addloglikelihood) : name_expr
        return AddLogLikelihood(addloglik_name, lower_to_wren(value_expr, mod))
    end

    # In order to keep the AST sensible, we preserve macros instead of expanding them -- but
    # we will lower their arguments. This means that something like
    #
    # @inbounds for i in 1:N
    #    x[i] ~ Normal()
    # end
    #
    # will be correctly handled. However, it does also mean that macros which expand into
    # new probabilistic statements are not supported. That's probably fine.
    #
    # TODO(penelopeysm): This limitation should be documented.
    if expr.head == :macrocall
        args = map(enumerate(expr.args)) do (index, arg)
            index <= 2 ? arg : lower_to_wren(arg, mod, statement_position)
        end
        return WrenExpr(:macrocall, args)
    end

    if expr.head == :call && expr.args[1] == :~
        length(expr.args) == 3 || error("@wren: expected ~ to have 2 arguments")
        lhs, rhs = expr.args[2:3]
        return Sample(Symbol(lhs), lhs, lower_to_wren(rhs, mod))
    end

    if expr.head == :return
        value = only(expr.args)
        return ReturnStatement(value === nothing ? nothing : lower_to_wren(value, mod))
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
        lower_to_wren(arg, mod, child_is_statement)
    end
    return WrenExpr(expr.head, args)
end

"""
    _wren_expr_to_expr(expr)

Unlower a Wren expression back into the way it was written in the model.
"""
_wren_expr_to_model_expr(expr) = expr
function _wren_expr_to_model_expr(stmt::WrenExpr)
    return Expr(stmt.head, map(_wren_expr_to_model_expr, stmt.args)...)
end
function _wren_expr_to_model_expr(stmt::Batch)
    body = _wren_expr_to_model_expr(stmt.body)
    for_expr = Expr(:for, Expr(:(=), stmt.loop_idx, stmt.loop_iters), body)
    args = Any[Symbol("@batch"), nothing]
    push!(args, Expr(:(=), :name, QuoteNode(stmt.name)))
    if stmt.size !== nothing
        push!(args, Expr(:(=), :size, stmt.size))
    end
    push!(args, for_expr)
    return Expr(:macrocall, args...)
end
_tilde_expr(stmt::{<:Union{Sample,Observe}}) = Expr(:call, :~, stmt.lhs, _wren_expr_to_model_expr(stmt.rhs))

function _wren_expr_to_model_expr(stmt::Sample)
    return _tilde_expr(stmt)
end
function _wren_expr_to_model_expr(stmt::Observe)
    return Expr(:macrocall, Symbol("@observe"), nothing, _tilde_expr(stmt))
end
function _wren_expr_to_model_expr(stmt::IID)
    args = Any[Symbol("@iid"), nothing, _tilde_expr(stmt.tilde)]
    if stmt.size !== nothing
        push!(args, Expr(:(=), :size, stmt.size))
    end
    iid_expr = Expr(:macrocall, args...)
    # An observed statement prints as `@observe @iid lhs ~ rhs size=dims`, with `@observe`
    # outermost: written the other way round, the inner `@observe` macrocall absorbs the
    # trailing `size=dims`, so the printed code no longer parses back to this node.
    stmt.tilde isa Observe || return iid_expr
    return Expr(:macrocall, Symbol("@observe"), nothing, iid_expr)
end
function _wren_expr_to_model_expr(stmt::AddLogLikelihood)
    args = Any[Symbol("@addloglikelihood"), nothing]
    if !startswith(string(stmt.name), "##addloglikelihood#")
        push!(args, Expr(:(=), :name, QuoteNode(stmt.name)))
    end
    push!(args, _wren_expr_to_model_expr(stmt.expr))
    return Expr(:macrocall, args...)
end
function _wren_expr_to_model_expr(stmt::ReturnStatement)
    value = stmt.expr === nothing ? nothing : _wren_expr_to_model_expr(stmt.expr)
    return Expr(:return, value)
end
_wren_expr_to_model_expr(stmt::OtherStatement) = stmt.expr

function _show_wren_exprs(io::IO, stmts, indent::Int=0; max_exprs::Int=typemax(Int))
    return _show_wren_exprs(io, stmts, ' '^indent; max_exprs)
end
function _show_wren_exprs(io::IO, stmts, prefix::String; max_exprs::Int=typemax(Int))
    n_shown = 0
    n_total = count(s -> s isa AbstractWrenExpr, stmts)
    indent = length(prefix)
    for stmt in stmts
        stmt isa AbstractWrenExpr || continue
        if n_shown >= max_exprs
            println(io)
            print(io, prefix, "... ($(n_total - max_exprs) more)")
            break
        end
        n_shown > 0 && println(io)
        print(io, prefix)
        _show_wren_expr(io, stmt, indent)
        n_shown += 1
    end
    return nothing
end
function _show_wren_expr(io::IO, stmt::AbstractWrenExpr, indent::Int=0)
    expr = deepcopy(_wren_expr_to_model_expr(stmt))
    expr isa Expr && Base.remove_linenums!(expr)
    return Base.show_unquoted(io, expr, indent, 0)
end

Base.show(io::IO, e::AbstractWrenExpr) = _show_wren_expr(io, e)
Base.show(io::IO, ::MIME"text/plain", e::AbstractWrenExpr) = _show_wren_expr(io, e)
function Base.show(
    io::IO,
    ::MIME"text/plain",
    stmts::AbstractVector{<:Union{AbstractWrenExpr,LineNumberNode}},
)
    println(io, "Wren expressions:")
    return _show_wren_exprs(io, stmts, 2)
end

end # module Expressions
