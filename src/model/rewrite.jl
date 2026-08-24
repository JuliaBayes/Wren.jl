"""
    rewrite(model::Model, (vn => value)::Pair...)::Model

Return a new model in which the right-hand side of the statement for
each `vn` is replaced by the runtime value `value`, exactly as if the
model had been written that way from the start. The original model is
untouched. Distributions and submodels keep the site a sampling
statement; any other value fixes the site to that deterministic value.
`vn` may name a plain site, a literal indexed site (`@vn(x[1])`), or a
plated or looped site family. `model[vn => value]` is equivalent sugar,
and `model[vn]` reads the current right-hand side. To splice in an
*expression* instead of a value, use [`@rewrite`](@ref).

```julia
julia> rewrite(f(), @vn(x) => LogNormal())  # replace the distribution

julia> f()[@vn(x) => 1.5]                   # fix the value of x
```

See the manual page on rewriting models for the full semantics, cost
model, and limitations.
"""
function rewrite(model::Model, ps::Pair{<:VarName}...)
    return foldl(_rewrite_pair, ps; init=model)
end

"""
    @rewrite model begin
        site ~ rhs
        other_site = value
    end

Return a new model in which the right-hand sides of the given sites are
replaced by the given expressions, exactly as if the model had been
written that way from the start. The original model is untouched. The
block shares the `@wren` model-body grammar and every statement must
target an existing site by name: `site ~ rhs` replaces a sampling
statement's right-hand side, `site = value` fixes the site to a
deterministic value, `@observe site ~ rhs` turns the statement into an
observation, and an allocation-and-construct pair (`site = zeros(N)`
followed by a `@plate` block or `for` loop) replaces the site with an
elementwise family. The spliced expressions become part of the model body,
so local variables at the `@rewrite` call site do not resolve; to
splice in a runtime value, use the function form [`rewrite`](@ref).

```julia
julia> f2 = @rewrite f() begin
           x ~ LogNormal()
       end;
```

See the manual page on rewriting models for the full semantics.
"""
macro rewrite(model_expr, patch_expr)
    patch = to_wren_exprs(patch_expr, __module__)
    _validate_patch(patch)
    return :($(_rewrite_patch)($(esc(model_expr)), $(patch)))
end

"""
    Base.getindex(model::Model, (vn => value)::Pair)::Model

Rewrite the model, replacing the right-hand side at the site `vn` with `value`.

`model[vn => value]` is an alias for [`rewrite(model, vn =>
value)`](@ref rewrite), by analogy with the mathematical substitution
``M[x := e]``.
"""
Base.getindex(model::Model, p::Pair{<:VarName}) = rewrite(model, p)

"""
    Base.getindex(model::Model, vn::VarName)

Return the current right-hand side value at the site `vn`.

`model[vn]` returns the distribution or submodel being sampled from at
the site `vn`, or the fixed value if the site has been made
deterministic with [`rewrite`](@ref).

The right-hand side expression is evaluated using the model's arguments.
It must therefore not depend on other variables defined in the model
body; if it does, an error is thrown.
"""
Base.getindex(model::Model, vn::VarName) = getsite(model, vn)

## Generic traversal over the Wren IR.

# `WrenExpr`s have their arguments as children, `Plate`s have their body
# (visited only when `plate_bodies` is set), and every other node is a
# leaf.
# TODO: this could be promoted to the Expressions module for reuse.
_stmt_children(stmt, ::Bool) = ()
_stmt_children(stmt::WrenExpr, ::Bool) = stmt.args
_stmt_children(stmt::Plate, plate_bodies::Bool) = plate_bodies ? (stmt.body,) : ()

"""
    _fold_stmts(f, acc, stmts; plate_bodies=true)

Fold `f(acc, stmt)` depth-first over every IR node in the statement (or
vector of statements) `stmts`, skipping `LineNumberNode`s.
"""
function _fold_stmts(f, acc, stmts::AbstractVector; plate_bodies::Bool=true)
    for stmt in stmts
        acc = _fold_stmts(f, acc, stmt; plate_bodies)
    end
    return acc
end
_fold_stmts(f, acc, lnn::LineNumberNode; plate_bodies::Bool=true) = acc
function _fold_stmts(f, acc, stmt; plate_bodies::Bool=true)
    acc = f(acc, stmt)
    for child in _stmt_children(stmt, plate_bodies)
        acc = _fold_stmts(f, acc, child; plate_bodies)
    end
    return acc
end

"""
    _map_stmts(f, stmts, count::Ref{Int}; plate_bodies=true)

Rebuild the statement (or vector of statements) `stmts`, replacing each
IR node for which `f(stmt)` returns a non-`nothing` value by that value
(without descending into the replacement), and incrementing `count` per
replacement. Nodes for which `f` returns `nothing` are rebuilt from
their transformed children.
"""
function _map_stmts(
    f,
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    count::Ref{Int};
    plate_bodies::Bool=true,
)
    return Union{AbstractWrenExpr,LineNumberNode}[
        _map_stmts(f, stmt, count; plate_bodies) for stmt in stmts
    ]
end
_map_stmts(f, lnn::LineNumberNode, ::Ref{Int}; plate_bodies::Bool=true) = lnn
function _map_stmts(f, stmt, count::Ref{Int}; plate_bodies::Bool=true)
    replacement = f(stmt)
    if replacement !== nothing
        count[] += 1
        return replacement
    end
    if stmt isa WrenExpr
        args = map(a -> _map_stmts(f, a, count; plate_bodies), stmt.args)
        return WrenExpr(stmt.head, args)
    elseif stmt isa Plate && plate_bodies
        body = _map_stmts(f, stmt.body, count; plate_bodies)
        return Plate(stmt.name, stmt.loop_idx, stmt.loop_iters, stmt.size, body)
    end
    return stmt
end

## Site naming and statement matching.

"""
    _lhs_name(lhs)

Canonical site name for a statement left-hand side, matching the
`Symbol(lhs)` convention used by `Sample` and `Observe` nodes. For
example, both `:x` and the expression `x[1]` have well-defined names.
"""
_lhs_name(lhs::Symbol) = lhs
_lhs_name(lhs) = Symbol(string(_wren_expr_to_model_expr(lhs)))

"""
    _site_name(vn::VarName)

The site name that `vn` targets, matching the `Symbol(lhs)` convention
used by the IR nodes: `@vn(x)` targets `x` and `@vn(x[1])` targets
`Symbol("x[1]")`.
"""
function _site_name(vn::VarName)
    optic = VarNames.getoptic(vn)
    optic isa VarNames.Iden && return VarNames.getsym(vn)
    return Symbol(string(vn))
end

"""
    _site_names(stmts)

Names of the sampling and observation sites in `stmts`, for error
messages.
"""
function _site_names(stmts)
    return _fold_stmts(Symbol[], stmts) do names, stmt
        stmt isa Union{Sample,Observe} ? push!(names, stmt.name) : names
    end
end

"""
    _is_site_assignment(stmt, name::Symbol)

Whether `stmt` is a plain assignment `name = ...` (as produced by a
do-style rewrite, or written in the model body).
"""
_is_site_assignment(stmt, ::Symbol) = false
function _is_site_assignment(stmt::WrenExpr, name::Symbol)
    return stmt.head === :(=) && length(stmt.args) == 2 && _lhs_name(stmt.args[1]) === name
end

"""
    _has_tilde_family(stmts, family::Symbol)

Whether any tilde statement (including inside loops and `@plate`
bodies) has `family` as the top-level symbol of its left-hand side.
"""
function _has_tilde_family(stmts, family::Symbol)
    return _fold_stmts(false, stmts) do found, stmt
        found ||
            (stmt isa Union{Sample,Observe} && get_top_level_symbol(stmt.lhs) === family)
    end
end

_is_family_mode(mode::Symbol) = mode === :family || mode === :family_observe
function _tilde_matches(stmt, name::Symbol, mode::Symbol)
    return if _is_family_mode(mode)
        get_top_level_symbol(stmt.lhs) === name
    else
        stmt.name === name
    end
end

"""
    _stmt_match(stmt, name::Symbol, replacement, mode::Symbol)

The replacement node for `stmt` if it is a target of the current pass,
and `nothing` otherwise. `mode` controls what counts as a match.
`:tilde` matches `Sample` statements by exact name and `:tilde_observe`
additionally matches `Observe` statements; `:family` and
`:family_observe` instead match tilde statements whose left-hand side
has `name` as its top-level symbol (e.g. `y[i]` inside a loop or
`@plate` body when targeting `y`); `:assignment` matches plain
assignments `name = ...` (as left behind by a do-style rewrite).
"""
function _stmt_match(stmt, name::Symbol, replacement, mode::Symbol)
    if stmt isa Sample
        (mode !== :assignment && _tilde_matches(stmt, name, mode)) || return nothing
        return replacement(stmt, _is_family_mode(mode))
    elseif stmt isa Observe
        ok =
            (mode === :tilde_observe || mode === :family_observe) &&
            _tilde_matches(stmt, name, mode)
        return ok ? replacement(stmt, _is_family_mode(mode)) : nothing
    elseif stmt isa WrenExpr && mode === :assignment && _is_site_assignment(stmt, name)
        return replacement(stmt, false)
    end
    return nothing
end

function _rewrite_pass(
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    name::Symbol,
    replacement,
    mode::Symbol,
)
    count = Ref(0)
    new_stmts = _map_stmts(stmts, count; plate_bodies=_is_family_mode(mode)) do stmt
        _stmt_match(stmt, name, replacement, mode)
    end
    return count[] == 0 ? nothing : new_stmts
end

"""
    _rewrite_stmts(
        stmts, name::Symbol, family::Symbol, replacement, match_observe::Bool;
        family_rhs=nothing, family_kind=:sample,
    )

Replace the statements for the site `name` in `stmts` with
`replacement(stmt, via_family)`. Exact-name tilde statements are
matched first, then whole-family targets, and lastly plain assignments
`name = ...` (which allow round-tripping do-style rewrites). The
assignment pass is never reached while tilde statements for the family
exist, so a family target cannot hijack an unrelated assignment such
as an array allocation. Errors if no statement matches, naming the
available sites.

A family target replaces the family's whole `@plate` or `for` construct
with the single statement `family ~ family_rhs` (`family_kind ===
:sample`) or `family = family_rhs` (`family_kind === :fixed`), exactly
as-if-written (see [`_collapse_family_stmts`](@ref)). The one exception
is a `:sample` right-hand side that references the construct's loop
index, which can only mean the elementwise statement, and therefore
replaces the right-hand side inside the construct instead.
"""
function _rewrite_stmts(
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    name::Symbol,
    family::Symbol,
    replacement,
    match_observe::Bool;
    family_rhs=nothing,
    family_kind::Symbol=:sample,
)
    result =
        _rewrite_pass(stmts, name, replacement, match_observe ? :tilde_observe : :tilde)
    result === nothing || return result
    if _has_tilde_family(stmts, family)
        if name !== family
            error(
                "rewrite: `$name` targets a single element of the site " *
                "family `$family`, which is not supported " *
                "(see https://github.com/JuliaBayes/Wren.jl/issues/19); " *
                "target the whole family `$family` instead.",
            )
        elseif family_rhs !== nothing && !(
            family_kind === :sample && _mentions_family_loop_idx(stmts, family, family_rhs)
        )
            new_stmt = if family_kind === :fixed
                WrenExpr(:(=), [family, family_rhs])
            else
                Sample(family, family, family_rhs)
            end
            return _collapse_family_stmts(
                stmts,
                family,
                Union{AbstractWrenExpr,LineNumberNode}[new_stmt],
            )
        end
        result = _rewrite_pass(
            stmts,
            name,
            replacement,
            match_observe ? :family_observe : :family,
        )
        result === nothing || return result
        # The family pass matches every statement that _has_tilde_family
        # counts, so this cannot be reached.
        _issue_error("rewrite: family pass failed to match for `$family`")
    end
    result = _rewrite_pass(stmts, name, replacement, :assignment)
    result === nothing || return result
    sites = join(unique!(_site_names(stmts)), ", ")
    return error("rewrite: no site named `$name` in the model (sites: $sites)")
end

## Collapsing a site family to a single statement.

# Whether `sym` occurs anywhere in an IR node or expression, including
# raw `Expr` leaves. Used as a conservative read/write check.
_mentions_symbol(x, sym::Symbol) = x === sym
_mentions_symbol(x::Expr, sym::Symbol) = any(a -> _mentions_symbol(a, sym), x.args)
_mentions_symbol(x::QuoteNode, sym::Symbol) = _mentions_symbol(x.value, sym)
_mentions_symbol(x::WrenExpr, sym::Symbol) = any(a -> _mentions_symbol(a, sym), x.args)
function _mentions_symbol(x::Union{Sample,Observe}, sym::Symbol)
    return _mentions_symbol(x.lhs, sym) || _mentions_symbol(x.rhs, sym)
end
function _mentions_symbol(x::Plate, sym::Symbol)
    return _mentions_symbol(x.loop_iters, sym) ||
           _mentions_symbol(x.size, sym) ||
           _mentions_symbol(x.body, sym)
end
_mentions_symbol(x::AddLogLikelihood, sym::Symbol) = _mentions_symbol(x.expr, sym)
function _mentions_symbol(x::ReturnStatement, sym::Symbol)
    return x.expr !== nothing && _mentions_symbol(x.expr, sym)
end
_mentions_symbol(x::OtherStatement, sym::Symbol) = _mentions_symbol(x.expr, sym)

_is_family_construct(stmt, ::Symbol) = false
_is_family_construct(stmt::Plate, family::Symbol) = _has_tilde_family(stmt, family)
function _is_family_construct(stmt::WrenExpr, family::Symbol)
    return stmt.head === :for && _has_tilde_family(stmt, family)
end

_construct_body(stmt::Plate) = stmt.body.args
_construct_body(stmt::WrenExpr) = stmt.args[2].args
_construct_loop_idx(stmt::Plate) = stmt.loop_idx
_construct_loop_idx(stmt::WrenExpr) = stmt.args[1].args[1]::Symbol
function _rebuild_construct(stmt::Plate, body)
    return Plate(
        stmt.name,
        stmt.loop_idx,
        stmt.loop_iters,
        stmt.size,
        WrenExpr(:block, body),
    )
end
_rebuild_construct(stmt::WrenExpr, body) =
    WrenExpr(:for, [stmt.args[1], WrenExpr(:block, body)])

# Whether `rhs` references the loop index of any construct containing
# the family, in which case it can only mean the elementwise statement.
function _mentions_family_loop_idx(stmts, family::Symbol, rhs)
    return _fold_stmts(false, stmts) do found, stmt
        found || (
            _is_family_construct(stmt, family) &&
            _mentions_symbol(rhs, _construct_loop_idx(stmt))
        )
    end
end

"""
    _collapse_family_stmts(stmts, family::Symbol, new_stmts::Vector)

Replace the plated or looped site family `family` by the statement
sequence `new_stmts` (a single tilde statement or assignment for
`family` itself, or an allocation followed by a replacement construct),
exactly as if the model had been written with `new_stmts` in place of
the family. The family's tilde statements are removed from the `@plate`
or `for` construct (which is dropped entirely if nothing else remains),
and earlier assignments to `family` that become dead are removed too.
Conservatively errors whenever statement reordering could change
behaviour, i.e. when other statements read `family` before `new_stmts`
would run.
"""
function _collapse_family_stmts(
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    family::Symbol,
    new_stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
)
    out = Union{AbstractWrenExpr,LineNumberNode}[]
    for stmt in stmts
        if !(stmt isa LineNumberNode) && _is_family_construct(stmt, family)
            append!(out, _collapse_family_construct(stmt, family, new_stmts))
        else
            push!(out, stmt)
        end
    end
    k = findfirst(s -> s === first(new_stmts), out)
    k === nothing && error(
        "rewrite: cannot rewrite the site family `$family` as a single " *
        "statement because its statements are not directly inside a " *
        "@plate block or for loop",
    )
    # Earlier assignments to the family (such as its array allocation)
    # are dead once the replacement runs, so drop them; if anything in
    # between reads the family, collapsing would change behaviour, so
    # refuse.
    keep = trues(length(out))
    for j in 1:(k-1)
        s = out[j]
        s isa LineNumberNode && continue
        if _is_site_assignment(s, family)
            keep[j] = false
        elseif _mentions_symbol(s, family)
            error(
                "rewrite: cannot rewrite the site family `$family` as a " *
                "single statement because earlier statements read `$family`",
            )
        end
    end
    return out[keep]
end

# Drop the family's tilde statements from one construct, returning the
# hoisted replacement sequence followed by the reduced construct (or
# the sequence alone if the body had nothing else).
function _collapse_family_construct(construct, family::Symbol, new_stmts)
    remaining = Union{AbstractWrenExpr,LineNumberNode}[]
    seen = false
    for s in _construct_body(construct)
        if s isa LineNumberNode
            push!(remaining, s)
        elseif s isa Observe && get_top_level_symbol(s.lhs) === family
            error(
                "rewrite: cannot rewrite the observed family `$family` as a " *
                "single statement",
            )
        elseif s isa Sample && get_top_level_symbol(s.lhs) === family
            seen = true
        else
            if !seen && _mentions_symbol(s, family)
                error(
                    "rewrite: cannot rewrite the site family `$family` as a " *
                    "single statement because earlier statements in its " *
                    "loop read `$family`",
                )
            end
            push!(remaining, s)
        end
    end
    seen || error(
        "rewrite: cannot rewrite the site family `$family` as a single " *
        "statement because its statements are nested inside another statement",
    )
    all(s -> s isa LineNumberNode, remaining) && return copy(new_stmts)
    return vcat(new_stmts, _rebuild_construct(construct, remaining))
end

"""
    _expand_family_stmts(stmts, family::Symbol, new_stmts::Vector)

Replace the site for `family` by an allocation-and-construct sequence,
turning a single-statement site back into an elementwise family. The
target may be a plain tilde site named `family`, a fixed assignment to
`family`, or an existing family construct (which swaps one construct
for another).
"""
function _expand_family_stmts(
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    family::Symbol,
    new_stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
)
    # A model that already has a construct for the family gets the
    # construct swapped (its allocation is handled there); otherwise the
    # family's single statement is replaced by the sequence.
    if any(s -> !(s isa LineNumberNode) && _is_family_construct(s, family), stmts)
        return _collapse_family_stmts(stmts, family, new_stmts)
    end
    out = Union{AbstractWrenExpr,LineNumberNode}[]
    n = 0
    for stmt in stmts
        if stmt isa LineNumberNode
            push!(out, stmt)
        elseif stmt isa Observe && stmt.name === family
            error(
                "rewrite: cannot replace the observed statement `$family` " *
                "with a loop or @plate block of latent sites",
            )
        elseif (stmt isa Sample && stmt.name === family) ||
               _is_site_assignment(stmt, family)
            append!(out, new_stmts)
            n += 1
        else
            push!(out, stmt)
        end
    end
    n == 1 && return out
    n > 1 && error(
        "rewrite: more than one statement for `$family`; cannot replace " *
        "it with a construct",
    )
    _has_tilde_family(stmts, family) && error(
        "rewrite: the statements for `$family` are nested inside another " *
        "statement; cannot replace them with a construct",
    )
    sites = join(unique!(_site_names(stmts)), ", ")
    return error("rewrite: no site named `$family` in the model (sites: $sites)")
end

## Caches.

# Cache for rewritten statements and their evaluation functions. The
# RGF's type uniquely identifies its expression, so two models sharing
# an evaluation function share the rewritten one too: only the first
# rewrite of a given form at a given site pays the RGF construction
# cost. Keys are
#
#   (typeof(evaluation_fn), sym, kind)         for the function form
#   (typeof(evaluation_fn), objectid(patch))   for the macro form
#
# where the macro form's patch vector is created once per @rewrite
# call site, at macro expansion time.
const _REWRITE_CACHE = Dict{Any,Any}()
# Also guards _GETSITE_CACHE below.
const _REWRITE_CACHE_LOCK = ReentrantLock()

function _rewritten_model(build, model::Model, args::NamedTuple, key)
    stmts, evaluation_fn = lock(_REWRITE_CACHE_LOCK) do
        get!(build, _REWRITE_CACHE, key)
    end
    return Model(
        model.name,
        args,
        stmts,
        model.mod;
        conditioned=model.conditioned,
        evaluation_fn,
    )
end

## The macro form.

# Whether a patch statement is a `@plate` block or plain for loop, to
# be swapped in for the target family's site wholesale.
_is_construct_patch(stmt) = stmt isa Plate || (stmt isa WrenExpr && stmt.head === :for)

# The site families sampled inside a construct.
function _construct_families(stmt)
    fams = _fold_stmts(Symbol[], stmt) do acc, s
        s isa Union{Sample,Observe} ? push!(acc, get_top_level_symbol(s.lhs)) : acc
    end
    return unique!(fams)
end

_is_assignment_patch(stmt) = stmt isa WrenExpr && stmt.head === :(=)

"""
    _validate_patch(patch)

Check that every statement in a `@rewrite` patch is of a supported
form. Called at macro expansion time.
"""
function _validate_patch(patch)
    stmts = [stmt for stmt in patch if !(stmt isa LineNumberNode)]
    for (i, stmt) in enumerate(stmts)
        if _is_construct_patch(stmt)
            fams = _construct_families(stmt)
            length(fams) == 1 || error(
                "@rewrite: a replacement loop or @plate block must sample " *
                "a single site family (got: $(join(fams, ", ")))",
            )
            prev = i > 1 ? stmts[i-1] : nothing
            (
                prev !== nothing &&
                _is_assignment_patch(prev) &&
                _lhs_name(prev.args[1]) === only(fams)
            ) || error(
                "@rewrite: a replacement loop or @plate block for " *
                "`$(only(fams))` must be immediately preceded by the " *
                "family's array allocation, e.g. `$(only(fams)) = zeros(N)`",
            )
            continue
        end
        ok =
            stmt isa Sample ||
            stmt isa Observe ||
            (
                _is_assignment_patch(stmt) &&
                length(stmt.args) == 2 &&
                (
                    stmt.args[1] isa Symbol ||
                    (stmt.args[1] isa WrenExpr && stmt.args[1].head === :ref)
                )
            )
        ok || error(
            "@rewrite: unsupported statement `$stmt`. Statements must " *
            "be of the form `site ~ rhs`, `@observe site ~ rhs`, " *
            "`site = value`, or an allocation-and-loop pair.",
        )
    end
    return nothing
end

_patch_target(stmt::Sample) = stmt.name
_patch_target(stmt::Observe) = stmt.name
_patch_target(stmt::Plate) = only(_construct_families(stmt))
function _patch_target(stmt::WrenExpr)
    stmt.head === :for && return only(_construct_families(stmt))
    return _lhs_name(stmt.args[1])
end
_patch_family(stmt::Union{Sample,Observe}) = get_top_level_symbol(stmt.lhs)
_patch_family(stmt::Plate) = _patch_target(stmt)
function _patch_family(stmt::WrenExpr)
    stmt.head === :for && return _patch_target(stmt)
    return get_top_level_symbol(_wren_expr_to_model_expr(stmt.args[1]))
end

# The replacement for one matched model statement, given one patch
# statement. A `~` patch keeps the matched statement's flavour (latent
# sites stay latent, observed statements stay observed) and its
# left-hand side, and replaces only the right-hand side; an `@observe`
# patch makes the statement an observation; an assignment patch makes
# the site deterministic.
_patch_replacement(patch::Sample, matched::Sample, ::Bool) =
    Sample(matched.name, matched.lhs, patch.rhs)
_patch_replacement(patch::Sample, matched::Observe, ::Bool) =
    Observe(matched.name, matched.lhs, patch.rhs)
function _patch_replacement(patch::Sample, ::WrenExpr, ::Bool)
    return Sample(patch.name, patch.lhs, patch.rhs)
end
function _patch_replacement(patch::Observe, matched::Union{Sample,Observe}, ::Bool)
    return Observe(matched.name, matched.lhs, patch.rhs)
end
function _patch_replacement(patch::Observe, matched::WrenExpr, ::Bool)
    return Observe(patch.name, patch.lhs, patch.rhs)
end
# Assignment patches targeting a family are handled by the collapse
# path before the family pass, so `via_family` cannot be set here.
_patch_replacement(patch::WrenExpr, matched, ::Bool) = patch

# The collapse right-hand side and kind for one patch statement:
# a `~` patch collapses a family to `family ~ rhs`, an assignment patch
# to `family = rhs`, and an `@observe` patch stays elementwise.
_patch_family_rhs(stmt::Sample) = (stmt.rhs, :sample)
_patch_family_rhs(stmt::Observe) = (nothing, :sample)
_patch_family_rhs(stmt::WrenExpr) = (stmt.args[2], :fixed)

function _rewrite_patch(model::Model, patch::Vector{Union{AbstractWrenExpr,LineNumberNode}})
    # A previous function-form rewrite of a patched site may have
    # left a hidden argument behind; the patch replaces the statement
    # that referred to it, so drop it.
    stale = filter(
        argname -> haskey(model.args, argname),
        [_rewrite_arg_name(_patch_target(p)) for p in patch if !(p isa LineNumberNode)],
    )
    args = Base.structdiff(model.args, NamedTuple{(stale...,)})
    key = (typeof(model.evaluation_fn), objectid(patch))
    return _rewritten_model(model, args, key) do
        stmts = model.stmts
        items = [p for p in patch if !(p isa LineNumberNode)]
        i = 1
        while i <= length(items)
            p = items[i]
            if i < length(items) &&
               _is_assignment_patch(p) &&
               _is_construct_patch(items[i+1]) &&
               _lhs_name(p.args[1]) === _patch_target(items[i+1])
                # An allocation-and-construct pair replaces the family's
                # site with the two statements in order.
                construct = items[i+1]
                stmts = _expand_family_stmts(
                    stmts,
                    _patch_target(construct),
                    Union{AbstractWrenExpr,LineNumberNode}[p, construct],
                )
                i += 2
                continue
            end
            family_rhs, family_kind = _patch_family_rhs(p)
            stmts = _rewrite_stmts(
                stmts,
                _patch_target(p),
                _patch_family(p),
                (matched, via_family) -> _patch_replacement(p, matched, via_family),
                true;
                family_rhs,
                family_kind,
            )
            i += 1
        end
        fn = make_evaluation_fn((keys(args)...,), stmts, model.mod)
        (stmts, fn)
    end
end

## The function form.

"""
    _rewrite_arg_name(sym::Symbol)

Name of the hidden model argument used to store a rewritten right-hand
side value for the site `sym`.
"""
_rewrite_arg_name(sym::Symbol) = Symbol("#rewrite#", sym)

"""
    _rewrite_kind(rhs)

Classify a replacement right-hand side value. Submodels and anything
with a density (in the `DensityInterface.DensityKind` sense) remain
sampling statements (`:sample`); any other value makes the site
deterministic (`:fixed`).
"""
function _rewrite_kind(rhs)
    is_submodel(typeof(rhs)) && return :sample
    return if DensityInterface.DensityKind(rhs) isa DensityInterface.NoDensity
        :fixed
    else
        :sample
    end
end

# The left-hand side of a matched statement, as a raw expression
# suitable for storing in a `Sample` node or an assignment.
_matched_lhs(matched::Union{Sample,Observe}) = matched.lhs
_matched_lhs(matched::WrenExpr) = _wren_expr_to_model_expr(matched.args[1])

# The replacement for one matched model statement when a runtime value
# is stored in the hidden argument `argname`. The matched statement's
# left-hand side is preserved. Family targets are handled by the
# collapse path, so `via_family` cannot be set here.
function _value_replacement(matched, argname::Symbol, kind::Symbol, ::Bool)
    lhs = _matched_lhs(matched)
    kind === :fixed && return WrenExpr(:(=), [lhs, argname])
    return Sample(_lhs_name(lhs), lhs, argname)
end

"""
    _rewritten_form(stmts, argname::Symbol)

Return `:sample` if some statement samples from the hidden argument
`argname`, `:fixed` if some statement assigns from it, and `nothing`
otherwise.
"""
function _rewritten_form(stmts, argname::Symbol)
    return _fold_stmts(nothing, stmts) do form, stmt
        form !== nothing && return form
        if stmt isa Sample && stmt.rhs === argname
            :sample
        elseif stmt isa WrenExpr &&
               stmt.head === :(=) &&
               length(stmt.args) == 2 &&
               stmt.args[2] === argname
            :fixed
        else
            nothing
        end
    end
end

function _rewrite_pair(model::Model, p::Pair{<:VarName})
    vn, new_rhs = p
    name = _site_name(vn)
    family = VarNames.getsym(vn)
    argname = _rewrite_arg_name(name)
    kind = _rewrite_kind(new_rhs)
    new_args = merge(model.args, NamedTuple{(argname,)}((new_rhs,)))
    if haskey(model.args, argname) && _rewritten_form(model.stmts, argname) === kind
        # The site has already been rewritten into the required form,
        # so only the hidden argument's value needs replacing.
        return Model(
            model.name,
            new_args,
            model.stmts,
            model.mod;
            conditioned=model.conditioned,
            evaluation_fn=model.evaluation_fn,
        )
    end
    key = (typeof(model.evaluation_fn), name, kind)
    return _rewritten_model(model, new_args, key) do
        replacement =
            (matched, via_family) -> _value_replacement(matched, argname, kind, via_family)
        stmts = _rewrite_stmts(
            model.stmts,
            name,
            family,
            replacement,
            false;
            family_rhs=argname,
            family_kind=kind,
        )
        fn = make_evaluation_fn((keys(new_args)...,), stmts, model.mod)
        (stmts, fn)
    end
end

## Retrieving the current right-hand side of a site.

"""
    _find_sample_rhs(stmts, sym::Symbol)

Find the unique `Sample` statement for `sym` and return its right-hand
side (as Wren IR). Errors if there is no match or more than one match.
"""
function _find_sample_rhs(stmts, sym::Symbol)
    found, n = _fold_stmts((nothing, 0), stmts; plate_bodies=false) do acc, stmt
        stmt isa Sample && stmt.name === sym ? (stmt.rhs, acc[2] + 1) : acc
    end
    n == 0 && error("no tilde-statement found for `$sym`")
    n > 1 && error("more than one tilde-statement found for `$sym`")
    return found
end

const _GETSITE_CACHE = Dict{Any,Any}()

# Internal implementation of `model[vn]`; see the docstring on
# `Base.getindex(::Model, ::VarName)`.
function getsite(model::Model, vn::VarName)
    sym = _site_name(vn)
    argname = _rewrite_arg_name(sym)
    # Fast path: the site was previously rewritten with a value, which
    # is stored directly in the hidden argument.
    haskey(model.args, argname) && return model.args[argname]
    key = (typeof(model.evaluation_fn), sym)
    fn = lock(_REWRITE_CACHE_LOCK) do
        get!(_GETSITE_CACHE, key) do
            rhs = _find_sample_rhs(model.stmts, sym)
            rhs_expr = _wren_expr_to_model_expr(rhs)
            _make_rhs_fn((keys(model.args)...,), rhs_expr, model.mod)
        end
    end
    return try
        fn(model.args)
    catch e
        e isa UndefVarError ?
        error(
            "getindex: the right-hand side of `$sym` depends on `$(e.var)`, " *
            "which is not a model argument, so its value cannot be " *
            "determined without running the model",
        ) : rethrow()
    end
end

function _make_rhs_fn(argnames::Tuple{Vararg{Symbol}}, rhs_expr, mod::Module)
    body = Expr(:block)
    @gensym model_args
    for k in argnames
        push!(body.args, :($(k) = $model_args.$k))
    end
    push!(body.args, rhs_expr)
    return RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
        mod,
        mod,
        :(function ($model_args,)
            $body
        end),
    )
end

"""
    _submodel_site_value(model::Model, sym::Symbol)

Return the submodel at the site `sym`, or `nothing`.

`nothing` is returned when the site is not a submodel, and also when
the site's right-hand side cannot be evaluated from the model's
arguments alone.
"""
function _submodel_site_value(model::Model, sym::Symbol)
    val = try
        getsite(model, VarName{sym}())
    catch
        return nothing
    end
    return is_submodel(typeof(val)) ? val : nothing
end

## Plate/loop conversion and argument replacement.

"""
    remove_plate(model::Model, [name::Symbol])::Model

Return a new model in which the named `@plate` block is replaced by the
equivalent plain `for` loop (same loop specification, same body),
exactly as if the model had been written with the loop. When the model
has exactly one `@plate` block, `name` may be omitted. The original
model is not modified.
"""
function remove_plate(model::Model, name::Union{Symbol,Nothing}=nothing)
    plates = _fold_stmts(Plate[], model.stmts) do acc, stmt
        stmt isa Plate ? push!(acc, stmt) : acc
    end
    isempty(plates) && error("remove_plate: the model has no @plate blocks")
    plate = if name === nothing
        length(plates) == 1 || error(
            "remove_plate: the model has $(length(plates)) @plate blocks; " *
            "pass the plate's name (available: " *
            "$(join([p.name for p in plates], ", ")))",
        )
        only(plates)
    else
        i = findfirst(p -> p.name === name, plates)
        i === nothing && error(
            "remove_plate: no @plate block named `$name` (available: " *
            "$(join([p.name for p in plates], ", ")))",
        )
        plates[i]
    end
    key = (typeof(model.evaluation_fn), :remove_plate, plate.name)
    return _rewritten_model(model, model.args, key) do
        loop_spec = WrenExpr(:(=), [plate.loop_idx, plate.loop_iters])
        stmts = _map_stmts(model.stmts, Ref(0)) do stmt
            stmt === plate ? WrenExpr(:for, [loop_spec, plate.body]) : nothing
        end
        (stmts, make_evaluation_fn((keys(model.args)...,), stmts, model.mod))
    end
end

"""
    add_plate(model::Model, vn::VarName; name::Symbol=...)::Model

Return a new model in which the plain `for` loop sampling the site
family `vn` is wrapped into a `@plate` block (with a generated name
unless `name` is given), exactly as if the model had been written with
`@plate`. The original model is not modified.
"""
function add_plate(model::Model, vn::VarName; name::Union{Symbol,Nothing}=nothing)
    family = _site_name(vn)
    key = (typeof(model.evaluation_fn), :add_plate, family, name)
    return _rewritten_model(model, model.args, key) do
        plate_name = name === nothing ? gensym(:plate) : name
        count = Ref(0)
        stmts = _map_stmts(model.stmts, count; plate_bodies=false) do stmt
            (stmt isa WrenExpr && stmt.head === :for && _has_tilde_family(stmt, family)) || return nothing
            body = stmt.args[2]
            for s in body.args
                s isa Union{LineNumberNode,Sample,Observe} || error(
                    "add_plate: the loop for `$family` contains statements " *
                    "other than tilde statements, which @plate does not " *
                    "support",
                )
            end
            return Plate(
                plate_name,
                _construct_loop_idx(stmt),
                _wren_expr_to_model_expr(stmt.args[1].args[2]),
                nothing,
                body,
            )
        end
        count[] == 1 || error(
            count[] == 0 ? "add_plate: no for loop samples the site family `$family`" :
            "add_plate: more than one for loop samples the site family `$family`",
        )
        (stmts, make_evaluation_fn((keys(model.args)...,), stmts, model.mod))
    end
end

"""
    set_arg(model::Model, name::Symbol, value)::Model

Return a new model with the argument `name` replaced by `value`. The
model's statements and evaluation function are reused unchanged, so no
code generation takes place. The original model is not modified.
"""
function set_arg(model::Model, name::Symbol, value)
    haskey(model.args, name) || error(
        "set_arg: no argument named `$name` (arguments: " *
        "$(join(keys(_visible_args(model)), ", ")))",
    )
    new_args = merge(model.args, NamedTuple{(name,)}((value,)))
    return Model(
        model.name,
        new_args,
        model.stmts,
        model.mod;
        conditioned=model.conditioned,
        evaluation_fn=model.evaluation_fn,
    )
end

## As-written display of rewritten models.

_is_rewrite_argname(k::Symbol) = startswith(String(k), "#rewrite#")

"""
    _visible_args(model::Model)

The model's arguments with hidden `#rewrite#` arguments removed, for
display purposes.
"""
function _visible_args(model::Model)
    ks = filter(!_is_rewrite_argname, keys(model.args))
    return NamedTuple{ks}(model.args)
end

"""
    _InlineValue

Display wrapper for a runtime value spliced into a statement's
right-hand side for showing a rewritten model as-written. Shows as the
value itself, and is deepcopy-safe even for values (such as models)
that cannot be deepcopied.
"""
struct _InlineValue{T}
    value::T
end
Base.show(io::IO, v::_InlineValue) = show(io, v.value)
Base.deepcopy_internal(v::_InlineValue, ::IdDict) = v

"""
    _display_stmts(model::Model)

The model's statements with hidden `#rewrite#` argument references
replaced by their stored values, so that a rewritten model displays as
if it had been written with those right-hand sides from the start.
"""
function _display_stmts(model::Model)
    any(_is_rewrite_argname, keys(model.args)) || return model.stmts
    return _map_stmts(model.stmts, Ref(0)) do stmt
        _display_match(stmt, model.args)
    end
end

# Rebuild a statement whose right-hand side references a hidden
# `#rewrite#` argument so that it shows the stored value inline.
function _display_match(stmt, args::NamedTuple)
    if stmt isa Sample
        rhs = _display_rhs(stmt.rhs, args)
        rhs === stmt.rhs ? nothing : Sample(stmt.name, stmt.lhs, rhs)
    elseif stmt isa Observe
        rhs = _display_rhs(stmt.rhs, args)
        rhs === stmt.rhs ? nothing : Observe(stmt.name, stmt.lhs, rhs)
    elseif stmt isa WrenExpr && stmt.head === :(=) && length(stmt.args) == 2
        rhs = _display_rhs(stmt.args[2], args)
        rhs === stmt.args[2] ? nothing : WrenExpr(:(=), [stmt.args[1], rhs])
    else
        nothing
    end
end
_display_rhs(rhs, ::NamedTuple) = rhs
function _display_rhs(rhs::Symbol, args::NamedTuple)
    return if _is_rewrite_argname(rhs) && haskey(args, rhs)
        _InlineValue(args[rhs])
    else
        rhs
    end
end
