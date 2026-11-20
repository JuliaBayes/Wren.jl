using Accessors: Accessors
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

"""
    WrenCore.Model

A Wren model.

$(TYPEDFIELDS)
"""
struct Model{A<:NamedTuple,C<:NamedTuple,F,CA<:Val}
    "Name of the function declared with @wren"
    name::Symbol
    "Positional and keyword arguments passed to the model."
    args::A
    "The model's AST, stored as Wren's probabilistic IR"
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}}
    "Module that the model was defined in. Required to make RGF resolve globals / functions
    defined in that module."
    mod::Module
    "The set of variables that the model is conditioned on."
    conditioned::C
    "The cached evaluation function."
    evaluation_fn::F
    "The names of the arguments which are copied on every evaluation when differentiating
    with Enzyme. This must be a `Val` of a tuple of symbols (preferably sorted to
    deduplicate compilation).

    Whether they are actually copied is decided by the [EvalContext](@ref) that the model is
    evaluated with. That struct allows us to differentiate code being run with Enzyme (where
    the copy is needed) from other code (where the copy is not needed)."
    enzyme_copy_argnames::CA

    function Model(
        name::Symbol,
        args::NamedTuple,
        stmts::Vector{<:Union{AbstractWrenExpr,LineNumberNode}},
        mod::Module;
        enzyme_copy_argnames::Val=Val{sort(keys(args))}(),
        conditioned::NamedTuple=NamedTuple(),
        _unsafe_reuse_evaluation_fn=nothing,
        _unsafe_skip_dealias::Bool=false,
    )
        # Copy arguments upon model construction.
        dealiased_args = _unsafe_skip_dealias ? args : map(_dealias, args)
        # Regenerate the evaluation function if it wasn't provided.
        evaluation_fn = if _unsafe_reuse_evaluation_fn === nothing
            make_evaluation_fn(keys(args), stmts, mod)
        else
            _unsafe_reuse_evaluation_fn
        end
        return new{
            typeof(dealiased_args),
            typeof(conditioned),
            typeof(evaluation_fn),
            typeof(enzyme_copy_argnames),
        }(
            name,
            dealiased_args,
            stmts,
            mod,
            conditioned,
            evaluation_fn,
            enzyme_copy_argnames,
        )
    end
end

"""
    _dealias

Create a copy of a model argument that has the same contents, but avoids aliasing the user's
data.

This is called on model construction, as well as on model evaluation (unless the argument
has been proven or annotated to not require copying). See [the docs](@ref enzyme-copyargs)
for why this matters.
"""
function _dealias(v::SubArray{T}) where {T}
    v = collect(v)  # Avoid copying the entire parent array.
    return _can_shallow_copy_elements(T) ? v : deepcopy(v)
end
function _dealias(v::AbstractArray{T}) where {T}
    return _can_shallow_copy_elements(T) ? copy(v) : deepcopy(v)
end
# For example, @NamedTuple{x::Union{Missing,Float64}} has an inline union field and
# is not a bits type. It has no GC-managed references, so deepcopy is unnecessary.
# Enzyme's reverse deepcopy rule does not support a Const input with a duplicated result.
_can_shallow_copy_elements(::Type{T}) where {T} =
    isbitstype(T) ||
    (isconcretetype(T) && !ismutabletype(T) && Base.datatype_pointerfree(T))
_dealias(v::T) where {T} = isbitstype(T) ? v : deepcopy(v)
# Models contain `Module`s, which can't be deepcopied by default (see the
# `Base.deepcopy_internal` method for `Model`). It is currently my belief that models
# should be immutable (or at least pretend to be), and thus we should not need to deepcopy
# models that are passed as arguments to models. However, this may need to be revisited.
_dealias(model::Model) = model

"""
    _needs_copy(::Type{T})

Whether a model argument of type `T` needs to be copied on every evaluation, assuming that
the dataflow analysis (or the user) has not ruled it out. The only reason for these copies
is to stop differentiable data being written into the model's arguments (see `_dealias`),
so types that can't hold differentiable data, or that `_dealias` returns unchanged anyway,
are never copied.

Note that this is separate from the copy made when the model is constructed, which protects
the user's data and always uses `_dealias`.
"""
_needs_copy(::Type{T}) where {T} = !isbitstype(T)
# Integers can't hold differentiable data: writing a float into an integer array either
# errors or creates a new array.
_needs_copy(::Type{<:AbstractArray{<:Integer}}) = false
_needs_copy(::Type{<:Model}) = false

"""
    _copied_args(::Model)

Get the set of argument names that are marked for copying. NOTE: This is for internal use
only. Be careful interpreting the results of this function! For example, even if
`_copied_args(model)` returns a nonempty set, this does not mean that the model's arguments
are actually copied. Outside of Enzyme contexts (i.e., in normal evaluation like
`parameters(model)`, the copies are always disabled. This includes AD with non-Enzyme backends.
And with Enzyme, it's very possible that [`reduce_copies`](@ref) will prune this set of
arguments down.
"""
_copied_args(::Model{A,C,F,Val{CA}}) where {A,C,F,CA} = CA

"""
    _with_copied_args(model::Model, copy_argnames)

Return the same model, but with `copy_argnames` (an iterable of symbols) as its
`enzyme_copy_argnames`. This reuses the evaluation function, so no new code is generated.
The names are sorted so that the same set of arguments always gives the same type (as in the
`@wren` macro).
"""
function _with_copied_args(model::Model, copy_argnames)
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned=model.conditioned,
        enzyme_copy_argnames=Val(Tuple(sort!(collect(Symbol, copy_argnames)))),
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end

"""
    _copy_for_thread(model::Model)

Return the same model with its own copies of the arguments and conditioned values, so that
evaluations on different threads cannot write into each other's data. Outside Enzyme,
evaluation does not copy arguments (see `_maybe_dealias`). Models passed as arguments are
copied too. This reuses the evaluation function, so no new code is generated.
"""
function _copy_for_thread(model::Model)
    return Model(
        model.name,
        map(_copy_model_arg, model.args),
        model.stmts,
        model.mod;
        conditioned=deepcopy(model.conditioned),
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        # Safe as the model is not changed in any way.
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end
# The constructor copies all other arguments with `_dealias`.
_copy_model_arg(v) = v
_copy_model_arg(model::Model) = _copy_for_thread(model)

# The default `deepcopy` fails on models because the `mod` field is a `Module`. The module,
# the evaluation function, and the statements never change, so they are shared; the
# arguments and conditioned values are deepcopied. The constructor's `_dealias` step is
# skipped so that the copied arguments stay the same objects as those in `stackdict` (e.g.
# a `VecModel`'s log-density function refers to the model's arguments too).
function Base.deepcopy_internal(model::Model, stackdict::IdDict)
    haskey(stackdict, model) && return stackdict[model]::typeof(model)
    copied = Model(
        model.name,
        Base.deepcopy_internal(model.args, stackdict),
        model.stmts,
        model.mod;
        conditioned=Base.deepcopy_internal(model.conditioned, stackdict),
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
        _unsafe_skip_dealias=true,
    )
    stackdict[model] = copied
    return copied
end

"""
    EvalContext{Enzyme}

Information about how a model is being evaluated. It is passed to the model's evaluation
function, and from there on to any submodels, so that they are evaluated in the same way.

`EnzymeCopies` is a `Bool` that says whether we are running under Enzyme AD. In such a case,
the arguments in the model's `enzyme_copy_argnames` should be copied on evaluation (see
`_maybe_dealias`).
"""
struct EvalContext{Enzyme} end

"""
    _maybe_dealias(::EvalContext, ::Val{S}, ::Val{name}, v)

Return `_dealias(v)` if the following all hold true:

- `EvalContext` tells us that we are running under Enzyme AD, which is the only situation where
  copies need to be made for correctness;
- `name` is in `S`, the set of argument names which must be copied (`S` can be pruned either through
  Wren's dataflow analysis, or by [`reduce_copies`](@ref)); and
- `v` is a type that needs copying (see `_needs_copy`).

Otherwise, returns `v`.

The decision only depends on types, so should be resolvable at compile-time.

Apparently, making this a generated function means that it's impossible for e.g. extensions
to overload `_needs_copy`, as the generated function only has access to the methods of
`_needs_copy` that were present at the compilation world age.
"""
@inline _maybe_dealias(::EvalContext{true}, ::Val{S}, ::Val{name}, v::V) where {S,name,V} =
    (name in S && _needs_copy(V)) ? _dealias(v) : v
@inline _maybe_dealias(::EvalContext{false}, ::Val, ::Val, v) = v

# Reactant's branch dataflow ignores gensyms. Accumulators must remain visible to @trace.
_evaluation_fn_args() =
    ntuple(_ -> Symbol("wren_local_", replace(string(gensym()), '#' => '_')), 7)

function evaluation_fn_expr(
    argnames::Tuple{Vararg{Symbol}},
    stmts::Vector{<:Union{AbstractWrenExpr,LineNumberNode}};
    fn_args=_evaluation_fn_args(),
)
    body = Expr(:block)
    latent_source,
    parameter_space,
    accs,
    model_args,
    conditioned,
    enzyme_copy_argnames,
    ctx = fn_args
    retval = gensym()
    for k in argnames
        # Conditioning an argument replaces it, so the model sees one value for `k`.
        push!(
            body.args,
            :(
                $(k) = if $(haskey)($conditioned, $(QuoteNode(k)))
                    $(getproperty)($conditioned, $(QuoteNode(k)))
                else
                    $(WrenCore._maybe_dealias)(
                        $ctx,
                        $enzyme_copy_argnames,
                        Val($(QuoteNode(k))),
                        $model_args.$k,
                    )
                end
            ),
        )
    end

    model_body =
        emit_statements(stmts, accs, conditioned, latent_source, parameter_space, ctx)
    push!(body.args, :($retval = $model_body))
    push!(body.args, :(return ($retval, $accs)))
    return Expr(:function, Expr(:tuple, fn_args...), body)
end

function make_evaluation_fn(argnames, stmts, mod::Module)
    return make_evaluation_fn(evaluation_fn_expr(argnames, stmts), mod)
end

function make_evaluation_fn(expr::Expr, mod::Module)
    # Expand macros before RGF converts their closures to opaque closures.
    return RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
        mod,
        mod,
        macroexpand(mod, expr),
    )
end

# for debugging
function view_evaluation_fn(m::Model)
    return RuntimeGeneratedFunctions.get_expression(m.evaluation_fn)
end

## macro stuff

function get_function_name_and_args(signature::Expr)
    if signature.head == :call
        # Keyword arguments parse as a leading `:parameters` block. Julia applies defaults
        # and type annotations through the escaped signature, so only the names matter here.
        args = signature.args[2:end]
        if !isempty(args) && Meta.isexpr(args[1], :parameters)
            args = [args[2:end]; args[1].args]
        end
        names = map(args) do arg
            while Meta.isexpr(arg, (:kw, :(::)), 2)
                arg = arg.args[1]
            end
            arg
        end
        return (signature.args[1], names)
    elseif signature.head == :where
        return get_function_name_and_args(signature.args[1])
    else
        error("@wren: expected function signature")
    end
end

function _show_namedtuple_tree(io::IO, nt::NamedTuple, prefix::String)
    nkeys = length(nt)
    compact_io = IOContext(io, :compact => true, :limit => true)
    for (i, (k, v)) in enumerate(pairs(nt))
        last = i == nkeys
        connector = last ? "└─ " : "├─ "
        print(io, prefix, connector)
        printstyled(io, k; color=:red)
        print(io, " => ")
        show(compact_io, v)
        last || println(io)
    end
end

function Base.show(io::IO, model::Model)
    print(io, "Model ", model.name)
    if !isempty(model.args)
        print(io, " (args: ", join(keys(model.args), ","), ")")
    end
end

# In the REPL
function Base.show(io::IO, ::MIME"text/plain", model::Model)
    printstyled(io, "Model ", model.name; bold=true)
    println(io)
    prefix = get(io, :tree_prefix, "")
    max_exprs = get(io, :max_wren_exprs, 20)
    entries = (
        "arguments" => Base.structdiff(model.args, model.conditioned),
        "conditioned" => model.conditioned,
        "expressions" => model.stmts,
    )
    for (i, (label, value)) in enumerate(entries)
        last = i == length(entries)
        connector = last ? "└─ " : "├─ "
        child_prefix = prefix * (last ? "   " : "│  ")
        print(io, prefix, connector)
        printstyled(io, label, ""; bold=true)
        if label == "expressions"
            println(io)
            _show_wren_exprs(io, value, child_prefix * "  "; max_exprs)
        elseif label == "arguments" || label == "conditioned"
            if isempty(value)
                printstyled(io, " (none)"; color=:white)
            else
                println(io)
                _show_namedtuple_tree(io, value, child_prefix)
            end
        end
        last || println(io)
    end
    return nothing
end

function extract_args(args_expr::Expr)
    args_expr.head == :tuple || error("@wren: expected a tuple of arguments")
    if isempty(args_expr.args)
        :(NamedTuple()), Set{Symbol}()
    else
        new_expr = Expr(:tuple)
        syms = Set{Symbol}()
        for arg in args_expr.args
            arg isa Symbol || error("@wren: expected function argument to be a symbol")
            push!(new_expr.args, :($arg = $(esc(arg))))
            push!(syms, arg)
        end
        new_expr, syms
    end
end

macro wren(args...)
    expr = last(args)
    internal_to_ir = false
    for opt in args[1:(end-1)]
        # `internal_to_ir=true` is meant for internal testing only. It makes `@wren` return
        # a tuple of the probabilistic IR (before dataflow analysis) and the set of the
        # function's argument names, instead of defining a model.
        if Meta.isexpr(opt, :(=)) && opt.args[1] === :internal_to_ir && opt.args[2] isa Bool
            internal_to_ir = opt.args[2]
        else
            error("@wren: unexpected argument `$opt`")
        end
    end
    _wren(expr, __module__; internal_to_ir)
end

macro sample(args...)
    error("@sample must be used inside @wren")
end

macro observe(args...)
    error("@observe must be used inside @wren")
end

macro iid(args...)
    error("@iid must be used inside @wren")
end

macro batch(args...)
    error("@batch must be used inside @wren")
end

macro addloglikelihood(args...)
    error("@addloglikelihood must be used inside @wren")
end

"""
    @track name = expr
    @track name

Record the value of `expr` as the tracked quantity `name` in a `@wren` model.
`@track name` is short for `@track name = name`. `name` must be a single symbol.

`@track` must be a statement of the model body. It is not supported inside `@batch` or
inside functions defined in the model; we discourage using it inside `@submodel`.

If `name` is used later in the model body, `expr` is also assigned to `name` and runs on
every evaluation of the model. If `name` is used only by later tracked quantities, it
is assigned only when one of those tracked quantities is requested. Otherwise `name` is
not defined in the model body.

!!! warning
    If `name` is not used later in the model body and the tracked quantity is not
    requested, `expr` is not run at all.
    Note that this means side effects in `expr` do not happen!

    Relatedly, if `expr` assigns to a variable defined elsewhere in the model body,
    that assignment happens only when `name` is requested.
"""
macro track(args...)
    error("@track must be used inside a @wren model body")
end

function _wren(expr::Expr, mod::Module; internal_to_ir::Bool=false)
    # The short form `f(x) = body` has the same signature and body as the long form.
    if expr.head == :(=) && Meta.isexpr(expr.args[1], (:call, :where))
        expr = Expr(:function, expr.args...)
    end
    # Sanity check
    expr.head == :function || error("@wren: expected a function definition")
    length(expr.args) == 2 || error("@wren: expected function expr with 2 expr-args")

    signature, body = expr.args

    # Generate a new symbol for the model's body, since we want the original
    # symbol to directly create the `Model`.
    fname, fargs = get_function_name_and_args(signature)
    fname isa Symbol || error("@wren: expected function name to be a symbol")
    fargs_expr = Expr(:tuple, fargs...)

    model_args, model_arg_symbols = extract_args(fargs_expr)
    ir = to_wren_exprs(body, mod)
    internal_to_ir && return :(($ir, $model_arg_symbols))
    model_exprs, args_to_copy = run_dataflow_analysis(ir, model_arg_symbols)
    # Sorted so that the same set of arguments always gives the same type.
    enzyme_copy_argnames = Val(Tuple(sort!(collect(args_to_copy))))

    # At macro definition time, we generate a new symbol to store the evaluation function
    # corresponding to the AST *as defined in the macro*. It's possible that users can
    # instantiate a Model and modify it, thus necessitating a new evaluation function.
    # That can be handled separately. However, we don't want *every* call to `model()` to
    # generate a new RGF because it should always use the RGF that corresponds to the
    # macro's contents. This RGF is stored as a module-level constant in order to ensure
    # type stability.
    #
    # TODO(penelopeysm): I wonder if it would be possible to store the RGF in an IdDict, and
    # make the model constructor, i.e., `model()`, a generated function, so that the type
    # instability is limited to compile time. This would be kind of similar to the way
    # autodiff works: already-derived rules are stored in a cache and looked up at compile
    # time, which can be type unstable, but the resulting rule is type stable.
    evaluation_fn_name = gensym(Symbol(fname, :_evaluation_fn))

    # These two `eval`s need to be separate. The first `eval` calls `RGF.init()`, which
    # itself does the same trick that we do here by installing a new module-level const. See
    #
    # https://github.com/SciML/RuntimeGeneratedFunctions.jl/blob/73411b2dad5e37815e678b7f6733fb9303600170/src/RuntimeGeneratedFunctions.jl#L387
    #
    # That definition causes the world age to be incremented...
    mod.eval(:($(RuntimeGeneratedFunctions.init)($(QuoteNode(mod)))))

    # ...and because `make_evaluation_fn` generates an RGF, which uses that exact const that
    # `RGF.init` installed, we have to make sure to eval it in a newer world age so that we
    # have access to that binding.
    mod.eval(
        quote
            const $evaluation_fn_name = $make_evaluation_fn(
                ($(QuoteNode.(fargs)...),),
                $model_exprs,
                $(QuoteNode(mod)),
            )
        end,
    )

    # This is the model constructor: it just looks up the cached RGF.
    evaluation_fn_ref = GlobalRef(mod, evaluation_fn_name)
    function_body = :($(Model)(
        $(QuoteNode(fname)),
        $model_args,
        $model_exprs,
        @__MODULE__;
        # Safe as the cached RGF is always correct for a given model instantiation, i.e.
        # `f(arg1, arg2)` and `f(arg3, arg4)` should always use the same RGF.
        _unsafe_reuse_evaluation_fn=($evaluation_fn_ref),
        enzyme_copy_argnames=($enzyme_copy_argnames),
    ))
    function_expr = Expr(:function, esc(signature), function_body)

    return function_expr
end
