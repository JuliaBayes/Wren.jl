using Accessors: Accessors
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

"""
    WrenCore.Model

A Wren model.

$(TYPEDFIELDS)
"""
struct Model{A<:NamedTuple,C<:VarNamedTuple,F}
    "Name of the function declared with @wren"
    name::Symbol
    "Arguments passed to the model. TODO(penelopeysm): kwargs"
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

    function Model(
        name::Symbol,
        args::NamedTuple,
        stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
        mod::Module;
        conditioned::VarNamedTuple=VarNamedTuple(),
        evaluation_fn=make_evaluation_fn(keys(args), stmts, mod),
    )
        dealiased_args = map(_dealias, args)
        return new{typeof(dealiased_args),typeof(conditioned),typeof(evaluation_fn)}(
            name,
            dealiased_args,
            stmts,
            mod,
            conditioned,
            evaluation_fn,
        )
    end
end

"""
    _dealias

Create a copy of a model argument that has the same contents, but avoids aliasing the user's
data. This is called on the model's arguments each time the model is evaluated.

The main purpose of this is in fact to ensure correctness under AD with Enzyme.

Specifically, the call to `DI.value_and_gradient` in `VecModel` passes the model's arguments
as a `DI.Const`, which means that the model's arguments should never contain differentiable
data. This is true at the start of the model (the differentiable data is in the active
`params` argument), but if we then _write_ params into the model's arguments, then any
gradients with respect to the model's arguments (i.e., gradients with respect to the params)
will be lost.

This is best illustrated with an example. Consider:

```julia
@wren function my_model(x)
    x[1:4] ~ dist
end
```

In the `VecModel` evaluation, this is lowered to something like:

```julia
function lp(params, x)
    p14 = view(params, 1:4)
    x[1:4] = p14  # Writing differentiable data into `x`!
    return logdensityof(dist, x[1:4])
end
```

Sometimes Enzyme does in fact return the correct gradient, but this relies on an LLVM
optimisation and cannot be guaranteed. See
https://github.com/EnzymeAD/Enzyme.jl/issues/2344.
"""
function _dealias(v::SubArray{T}) where {T}
    v = collect(v)  # Avoid copying the entire parent array.
    return isbitstype(T) ? v : deepcopy(v)
end
function _dealias(v::AbstractArray{T}) where {T}
    return isbitstype(T) ? copy(v) : deepcopy(v)
end
_dealias(v) = deepcopy(v)
# Models contain `Module`s, which can't be deepcopied. It is currently my belief that models
# should be immutable (or at least pretend to be), and thus we should not need to deepcopy
# models that are passed as arguments to models. However, this may need to be revisited.
_dealias(model::Model) = model

function make_evaluation_fn(
    argnames::Tuple{Vararg{Symbol}},
    stmts::Vector{Union{AbstractWrenExpr,LineNumberNode}},
    mod::Module,
)
    body = Expr(:block)
    @gensym model_args
    for k in argnames
        push!(body.args, :($(k) = $(WrenCore._dealias)($model_args.$k)))
    end

    @gensym accs conditioned latent_source parameter_space retval
    model_body = emit_statements(stmts, accs, conditioned, latent_source, parameter_space)
    push!(body.args, :($retval = $model_body))
    push!(body.args, :(return ($retval, $accs)))
    return RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
        mod,
        mod,
        :(function ($latent_source, $parameter_space, $accs, $model_args, $conditioned)
            $body
        end),
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
                _show_namedtuple_tree(io, value, child_prefix)
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

function to_namedtuple(args_expr::Expr)
    args_expr.head == :tuple || error("@wren: expected a tuple of arguments")
    if isempty(args_expr.args)
        :(NamedTuple())
    else
        new_expr = Expr(:tuple)
        for arg in args_expr.args
            arg isa Symbol || error("@wren: expected function argument to be a symbol")
            push!(new_expr.args, :($arg = $(esc(arg))))
        end
        new_expr
    end
end

macro wren(expr)
    _wren(expr, __module__)
end

macro observe(expr)
    error("@observe must be used inside @wren")
end

macro iid(expr)
    error("@iid must be used inside @wren")
end

macro batch(args...)
    error("@batch must be used inside @wren")
end

macro addloglikelihood(args...)
    error("@addloglikelihood must be used inside @wren")
end

function _wren(expr::Expr, mod::Module)
    # Sanity check
    expr.head == :function || error("@wren: expected a function definition")
    length(expr.args) == 2 || error("@wren: expected function expr with 2 expr-args")

    # Generate a new symbol for the model's body, since we want the original
    # symbol to directly create the `Model`.
    fname, fargs = get_function_name_and_args(expr.args[1])
    fname isa Symbol || error("@wren: expected function name to be a symbol")
    fargs_expr = Expr(:tuple, fargs...)
    signature, body = expr.args

    model_args = to_namedtuple(fargs_expr)
    model_exprs = to_wren_exprs(body, mod)

    # At macro definition time, we generate a new symbol to store the evaluation function
    # corresponding to the AST *as defined in the macro*. It's possible that users can
    # instantiate a Model and modify it, thus necessitating a new evaluation function.
    # That's handled separately. However, we don't want *every* call to `model()` to
    # generate a new RGF because it should always use the RGF that corresponds to the
    # macro's contents.
    #
    # The cached RGF is stored as a module-level constant in order to ensure type stability.
    # We could also store it in an IdDict or something like that. That avoids the cost of
    # constructing the RGF, but is type unstable.
    #
    # TODO(penelopeysm): I wonder if it would be possible to store the RGF in an IdDict, and
    # make the model constructor, i.e., `model()`, a generated function, so that the type
    # instability is limited to compile time. This would be kind of similar to the way
    # autodiff works: already-derived rules are stored in a cache and looked up at compile
    # time, which can be type unstable, but the resulting rule is type stable.
    evaluation_fn_name = gensym(Symbol(fname, :_evaluation_fn))
    # These two `eval`s need to be separate because RGF.init() itself does the same
    # trick that we do here by installing a new module-level const. See
    #
    # https://github.com/SciML/RuntimeGeneratedFunctions.jl/blob/73411b2dad5e37815e678b7f6733fb9303600170/src/RuntimeGeneratedFunctions.jl#L387
    #
    # That causes the world age to be incremented...
    mod.eval(:($(RuntimeGeneratedFunctions.init)($(QuoteNode(mod)))))
    # ...and because `make_evaluation_fn` generates an RGF, which uses that const that RGF
    # installed, we have to make sure to eval it in a newer world age so that we have access
    # to the binding.
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
        evaluation_fn=($evaluation_fn_ref),
    ))
    function_expr = Expr(:function, esc(signature), function_body)

    return function_expr
end
