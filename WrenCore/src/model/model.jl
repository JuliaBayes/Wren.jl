using Accessors: Accessors
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

"""
    WrenCore.Model

A Wren model.

$(TYPEDFIELDS)
"""
struct Model{A<:NamedTuple,C<:VarNamedTuple,F,CA<:Val}
    "Name of the function declared with @wren"
    name::Symbol
    "Positional arguments passed to the model."
    args::A # TODO(penelopeysm): kwargs.
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
        stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
        mod::Module;
        enzyme_copy_argnames::Val,
        conditioned::VarNamedTuple=VarNamedTuple(),
        _unsafe_reuse_evaluation_fn=nothing,
    )
        # Copy arguments upon model construction.
        dealiased_args = map(_dealias, args)
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
    return isbitstype(T) ? v : deepcopy(v)
end
function _dealias(v::AbstractArray{T}) where {T}
    return isbitstype(T) ? copy(v) : deepcopy(v)
end
_dealias(v::T) where {T} = isbitstype(T) ? v : deepcopy(v)
# Models contain `Module`s, which can't be deepcopied. It is currently my belief that models
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
  Wren's dataflow analysis, or by a user's `@nocopy` annotations); and
- `v` is a type that needs copying (see `_needs_copy`).

Otherwise, returns `v`.

The decision only depends on types, so should be resolvable at compile-time.

Apparently, making this a generated function means that it's impossible for e.g. extensions
to overload `_needs_copy`, as the generated function only has access to the methods of
`_needs_copy` that were present at the compilation world age.
"""
@inline _maybe_dealias(::EvalContext{true}, ::Val{S}, ::Val{name}, v::V) where {S,name,V} =
    name in S && _needs_copy(V) ? _dealias(v) : v
@inline _maybe_dealias(::EvalContext{false}, ::Val, ::Val, v) = v

function make_evaluation_fn(
    argnames::Tuple{Vararg{Symbol}},
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    mod::Module,
)
    body = Expr(:block)
    @gensym model_args enzyme_copy_argnames ctx
    for k in argnames
        push!(
            body.args,
            :(
                $(k) = $(WrenCore._maybe_dealias)(
                    $ctx,
                    $enzyme_copy_argnames,
                    Val($(QuoteNode(k))),
                    $model_args.$k,
                )
            ),
        )
    end

    @gensym accs conditioned latent_source parameter_space retval
    model_body =
        emit_statements(stmts, accs, conditioned, latent_source, parameter_space, ctx)
    push!(body.args, :($retval = $model_body))
    push!(body.args, :(return ($retval, $accs)))
    return RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
        mod,
        mod,
        :(
            function (
                $latent_source,
                $parameter_space,
                $accs,
                $model_args,
                $conditioned,
                $enzyme_copy_argnames,
                $ctx,
            )
                $body
            end
        ),
    )
end

# for debugging
function view_evaluation_fn(m::Model)
    return RuntimeGeneratedFunctions.get_expression(m.evaluation_fn)
end

## macro stuff

function get_function_name_and_args(signature::Expr)
    if signature.head == :call
        return (signature.args[1], signature.args[2:end])
    elseif signature.head == :where
        return get_function_name_and_args(signature.args[1])
    else
        error("@wren: expected function signature")
    end
end

"""
    strip_nocopy_annotations(signature::Expr)

Remove `@nocopy` annotations from the arguments in a model's function signature. Returns the
new signature together with the set of argument names that were annotated.
"""
function strip_nocopy_annotations(signature::Expr)
    if signature.head == :where
        new_inner, nocopy_args = strip_nocopy_annotations(signature.args[1])
        return Expr(:where, new_inner, signature.args[2:end]...), nocopy_args
    end
    signature.head == :call || error("@wren: expected function signature")
    nocopy_args = Set{Symbol}()
    new_args = map(signature.args[2:end]) do arg
        # Only the bare `@nocopy` is recognised, not qualified forms like
        # `WrenCore.@nocopy`. Since this is a purely syntactic check, we can't tell what a
        # qualified name actually refers to.
        Meta.isexpr(arg, :macrocall) && arg.args[1] === Symbol("@nocopy") || return arg
        # `arg.args[2]` is a LineNumberNode.
        inner = arg.args[3:end]
        if !(length(inner) == 1 && inner[1] isa Symbol)
            error("@wren: `@nocopy` must be applied to a single argument name, got `$arg`")
        end
        push!(nocopy_args, only(inner))
        return only(inner)
    end
    return Expr(:call, signature.args[1], new_args...), nocopy_args
end

function _show_namedtuple_tree(io::IO, nt::NamedTuple, prefix::String, copied::Tuple)
    nkeys = length(nt)
    compact_io = IOContext(io, :compact => true, :limit => true)
    for (i, (k, v)) in enumerate(pairs(nt))
        last = i == nkeys
        connector = last ? "└─ " : "├─ "
        print(io, prefix, connector)
        printstyled(io, k; color=:red)
        # Arguments in the copy set are only really copied if `_needs_copy` says so.
        if k in copied && _needs_copy(typeof(v))
            printstyled(io, " (copied)"; color=:white)
        end
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
function Base.show(io::IO, ::MIME"text/plain", model::Model{A,C,F,Val{S}}) where {A,C,F,S}
    printstyled(io, "Model ", model.name; bold=true)
    println(io)
    prefix = get(io, :tree_prefix, "")
    max_exprs = get(io, :max_wren_exprs, 20)
    entries = (
        "arguments" => model.args,
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
        elseif label == "arguments"
            if isempty(value)
                printstyled(io, " (none)"; color=:white)
            else
                println(io)
                _show_namedtuple_tree(io, value, child_prefix, S)
            end
        elseif label == "conditioned" && value isa VarNamedTuple
            if isempty(value)
                printstyled(io, " (none)"; color=:white)
            else
                compact_io = IOContext(io, :compact => true, :limit => true)
                println(io)
                vnt_pretty_print(compact_io, value, child_prefix, 0; show_header=false)
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
            # TODO(penelopeysm): support default positional arguments or keyword arguments.
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

macro observe(expr)
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
    @nocopy

Mark a model argument as one that should never be copied when the model is evaluated, e.g.

```julia
@wren function f(x, @nocopy(y))
    ...
end
```

By default, Wren copies an argument on every evaluation if it cannot rule out that the
argument is written to and then read from again (see the [Enzyme docs](@ref
enzyme-copyargs) for why this matters). `@nocopy` overrides this.

It is only safe to use `@nocopy` on an argument if, inside the model, the argument is either
never written to, or is written to but never read from afterwards. If this does not hold,
gradients computed with Enzyme may be silently wrong.
"""
macro nocopy(args...)
    error("@nocopy must be used on an argument in the signature of a @wren function")
end

function _wren(expr::Expr, mod::Module; internal_to_ir::Bool=false)
    # Sanity check
    expr.head == :function || error("@wren: expected a function definition")
    length(expr.args) == 2 || error("@wren: expected function expr with 2 expr-args")

    # Remove `@nocopy` annotations from the signature, since the model constructor we emit
    # below should just have plain arguments.
    signature, body = expr.args
    signature, nocopy_args = strip_nocopy_annotations(signature)

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
    enzyme_copy_argnames = Val(Tuple(sort!(collect(setdiff(args_to_copy, nocopy_args)))))

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
