using ADTypes: NoAutoDiff # DI doesn't export
using EnzymeCore: Reverse

const AUTO_RUNTIME_ACTIVITY = Ref{Union{Nothing,Bool}}(true)

const DEFAULT_AD = DI.AutoEnzyme(; mode=Reverse)

"""
    WrenCore.set_auto_runtime_activity!(val::Union{Bool,Nothing})

Set Wren's automatic runtime activity behaviour.

When using Enzyme as the AD backend, depending on how the model is constructed, it may be
necessary to enable runtime activity. This can often allow more expressive models to be
differentiated, but comes at a performance cost.

Wren recommends using reverse-mode Enzyme, i.e., `AutoEnzyme(; mode=Reverse)` as the AD
backend, which does not enable runtime activity. **By default, Wren will automatically
enable runtime activity if Enzyme requires it.**

If you would rather have Wren raise an error, set

```julia
WrenCore.set_auto_runtime_activity!(false)
```

Or if you want an info message instead of an error, set

```julia
WrenCore.set_auto_runtime_activity!(nothing)
```

To restore the default behaviour (silently enabling runtime activity), set

```julia
WrenCore.set_auto_runtime_activity!(true)
```
"""
function set_auto_runtime_activity!(val::Union{Bool,Nothing})
    AUTO_RUNTIME_ACTIVITY[] = val
    return nothing
end

"""
    _VecModelLP

Callable struct that is used for AD backends that require a closure (i.e. compiled ReverseDiff).
This is equivalent to `_vecmodel_lp` but with the inactive arguments stored in the struct.
"""
struct _VecModelLP{
    E,
    A<:NamedTuple,
    C<:NamedTuple,
    CA<:Val,
    X<:EvalContext,
    F,
    S<:AbstractParameterSpace,
    R<:VecInfo,
    AC<:AccumulatorTuple,
}
    evaluation_fn::E
    model_args::A
    conditioned::C
    enzyme_copy_argnames::CA
    ctx::X
    lp_function::F
    parameter_space::S
    vecinfo::R
    accs::AC
end
function (f::_VecModelLP)(params::AbstractVector)
    return _vecmodel_lp(
        params,
        f.evaluation_fn,
        f.model_args,
        f.conditioned,
        f.enzyme_copy_argnames,
        f.ctx,
        f.lp_function,
        f.parameter_space,
        f.vecinfo,
        f.accs,
    )
end

"""
    _vecmodel_lp

The function that calculates the log-density at `params`.
"""
function _vecmodel_lp(
    params::AbstractVector,
    evaluation_fn,
    model_args::NamedTuple,
    conditioned::NamedTuple,
    enzyme_copy_argnames::Val,
    ctx::EvalContext,
    lp_function,
    parameter_space::AbstractParameterSpace,
    vecinfo::VecInfo,
    accs::AccumulatorTuple,
)
    source = FromVector(params, vecinfo)
    _, accs = evaluation_fn(
        source,
        parameter_space,
        accs,
        model_args,
        conditioned,
        enzyme_copy_argnames,
        ctx,
    )
    return lp_function(accs)
end

"""
    VecModel(
        model::Model,
        adtype::Union{ADTypes.AbstractADType,Nothing};
        lp_function=logjoint_transformed,
        latent_source=FromPrior(),
        parameter_space=UnconstrainedSpace(),
        reduce_copies=true,
    )

A vectorised form of `model`, which is designed to take a single vector of parameters as
input, and returns a scalar log-density.

## Overview

This is accomplished by storing the original model, together with a set of *ranges and
transforms* which map segments of the parameter vector to the latent variables' values in
original space. For example, consider

```julia
@wren function my_model()
    x ~ Normal()
    y ~ Normal()
end
```

The mapping from `params`, a vector of length 2 to the latent variables takes the form:

```julia
x_info = mapping[@vn(x)] # This is a `RangeAndTransform` object.
x = x_info.transform(params[x_info.range])

y_info = mapping[@vn(y)]
y = y_info.transform(params[y_info.range])
```

where `x_info.range` might be `1:1` and `y_info.range` might be `2:2`.

In order for this to work reliably, the model *must* have a fixed set of latent variables
and fixed dimension. If these conditions are not adhered to, it is very likely that results
obtained will be meaningless. Models with varying parameter *order* can be safely vectorised
since the mapping is invariant to the order in which the latent variables are encountered.

## Positional arguments

- `model::Model`: The model to vectorise.

- `adtype::Union{ADTypes.AbstractADType,Nothing}`: The automatic differentiation backend to
  use for gradient calculations. You can pass `nothing` to construct a log-density-only
  model that is not capable of computing gradients.

  In Wren, we recommend using reverse-mode Enzyme as the default backend as that offers the
  most performance. This can be specified with `AutoEnzyme(; mode=Reverse)` (these identifiers
  are re-exported by Wren).

  However, in some cases you may find that Enzyme errors. **In such a case, please do open
  an issue: we want to help fix this!** In such cases you can use a different AD backend.
  Wren is backed by
  [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) so
  you can plug in any `AbstractADType`.

## Keyword arguments

- `latent_source::AbstractLatentSource`: when preparing a `VecModel`, Wren requires
  information about the model's latent variables in order to construct the mapping from
  vector to latent variables. This is obtained by running the model once with an
  `AbstractLatentSource`, which you can pass here. If you want to control the RNG with which
  these parameters are drawn, you can use [`FromPrior(rng)`](@ref FromPrior). If you have a
  `VarNamedTuple` of values, wrap it in [`FromValues`](@ref).

  If you have already run the model once, you can avoid reevaluation entirely by calling the
  positional constructor with a `VecInfo` and a parameter vector. This is mostly
  intended for advanced usage, please see the Wren source code for examples of how to do
  this.

- `lp_function`: When the model is evaluated, it will return a set of accumulators, which must
  then be collapsed into a single scalar log-density. This function is responsible for that
  (i.e., it must have the signature `f(accs::AccumulatorTuple) -> Number`). The default
  is [`logjoint_transformed`](@ref), which evaluates the log joint density in transformed
  space. This is virtually always what you want, but you can also use [`logprior`](@ref),
  [`loglikelihood`](@ref), or [`logjoint`](@ref), or write your own.

- `parameter_space::AbstractParameterSpace`: The parameter space in which the vectorised
  model is defined. The default is [`UnconstrainedSpace`](@ref), which is almost always what
  you want (again this is mostly for compatibility with gradient-based inference algorithms).

- `reduce_copies::Bool`: When differentiating with Enzyme, Wren copies some of the model's
  arguments on every evaluation (see [the docs](@ref enzyme-copyargs)). If this is `true`,
  [`reduce_copies`](@ref) is first called on the model to find out which of these copies can
  be skipped. This has no effect for other AD backends, which never make the copies. The
  default is `true`.

  In very pathological cases, `reduce_copies` may skip a copy that is actually needed (e.g.
  when there is parameter-dependent control flow which (1) in turn causes a change in Enzyme's
  behaviour, and (2) is not caught by the random testing in `reduce_copies`). This is very
  unlikely, but if you are concerned about this, you can disable this option.
"""
struct VecModel{
    M<:Model,
    A<:Union{DI.AbstractADType,Nothing},
    F,
    S<:AbstractParameterSpace,
    R<:VecInfo,
    AC<:AccumulatorTuple,
    L<:_VecModelLP,
    P,
    X<:AbstractVector,
}
    model::M
    adtype::A
    lp_function::F
    parameter_space::S
    vecinfo::R
    accs::AC
    dimension::Int
    lp_func::L
    adprep::P
    sample::X
end

# This is the main constructor. All callers of VecModel should go through this constructor!
#
# Unlike in DynamicPPL, this has to be an outer constructor because the ReactantExt requires
# access to the direct (generated) constructor.
function VecModel(
    model::M,
    adtype::A,
    lp_function::F,
    parameter_space::S,
    vecinfo::R,
    params::X,
) where {
    M<:Model,
    A<:Union{DI.AbstractADType,Nothing},
    F,
    S<:AbstractParameterSpace,
    R<:VecInfo,
    X<:AbstractVector,
}
    # TODO: make accs customisable
    accs = AccumulatorTuple(
        LogPriorAccumulator(),
        LogJacobianAccumulator(),
        LogLikelihoodAccumulator(),
    )
    lp_func = _VecModelLP(
        model.evaluation_fn,
        model.args,
        model.conditioned,
        model.enzyme_copy_argnames,
        # Safe to avoid the copy because for Enzyme (where the copy is needed) we will call
        # `DI.prepare_gradient` with the correct EvalContext below.
        EvalContext{false}(),
        lp_function,
        parameter_space,
        vecinfo,
        accs,
    )
    adprep = if adtype === nothing || adtype isa NoAutoDiff
        nothing
    elseif _use_closure(adtype)
        # Compiled ReverseDiff mandates the closure, otherwise preparation is repeated on
        # each evaluation.
        DI.prepare_gradient(lp_func, adtype, params)
    else
        # Other AD backends can directly take multiple arguments, which can often be
        # slightly better for performance.
        DI.prepare_gradient(
            _vecmodel_lp,
            adtype,
            params,
            DI.Constant(model.evaluation_fn),
            DI.Constant(model.args),
            DI.Constant(model.conditioned),
            DI.Constant(model.enzyme_copy_argnames),
            DI.Constant(_eval_context(adtype)),
            DI.Constant(lp_function),
            DI.Constant(parameter_space),
            DI.Constant(vecinfo),
            DI.Constant(accs),
        )
    end
    vm = VecModel(
        model,
        adtype,
        lp_function,
        parameter_space,
        vecinfo,
        accs,
        length(params),
        lp_func,
        adprep,
        params,
    )
    return _handle_enzyme(vm)
end

# Convenience constructor that calls the main one.
function VecModel(
    model::Model,
    adtype::Union{DI.AbstractADType,Nothing};
    latent_source::AbstractLatentSource=FromPrior(),
    lp_function=logjoint_transformed,
    parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
    reduce_copies::Bool=true,
)
    model = if reduce_copies
        WrenCore.reduce_copies(
            model,
            adtype;
            lp_function,
            latent_source,
            parameter_space,
            verbose=false,
        )
    else
        model
    end
    vecinfo, sample = _prepare_vecmodel_info(model, latent_source, parameter_space)
    return VecModel(model, adtype, lp_function, parameter_space, vecinfo, sample)
end

function Base.show(io::IO, ::MIME"text/plain", vm::VecModel)
    printstyled(io, "VecModel ", vm.model.name; bold=true)
    println(io)
    entries = Any[
        "adtype"=>vm.adtype,
        "dimension"=>vm.dimension,
        "parameter space"=>vm.parameter_space,
        "model"=>vm.model,
    ]
    for (i, (label, value)) in enumerate(entries)
        last = i == length(entries)
        connector = last ? "└─ " : "├─ "
        child_prefix = last ? "   " : "│  "
        print(io, connector)
        if label == "model"
            show(
                IOContext(io, :max_wren_exprs => 10, :tree_prefix => child_prefix),
                MIME"text/plain"(),
                value,
            )
        else
            printstyled(io, label; bold=true)
            print(io, " => ")
            show(io, value)
        end
        last || println(io)
    end
    return nothing
end

_use_closure(::DI.AutoReverseDiff{compile}) where {compile} = compile
_use_closure(::DI.AbstractADType) = false

"""
    _eval_context(adtype)

The [`EvalContext`](@ref) that a `VecModel` with the given AD backend should evaluate its
model with. Only Enzyme needs model arguments to be copied on evaluation (see [the
docs](@ref enzyme-copyargs)); other AD backends can track gradients through writes into
model arguments, so the copies can be skipped.
"""
_eval_context(::Union{DI.AbstractADType,Nothing}) = EvalContext{false}()
_eval_context(::DI.AutoEnzyme) = EvalContext{true}()

_copied_args(vm::VecModel) = _copied_args(vm.model)

# overloaded in EnzymeExt as it depends on runtime activity error which isn't in Core
_handle_enzyme(@nospecialize(vm::VecModel)) = vm

# Create a copy of a VecModel with a deepcopied adprep. This is necessary for multithreaded
# sampling because AD preparation objects are not thread-safe.
# https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/faq/limitations/#Multithreading
function _copy_for_thread(vm::VecModel)
    return VecModel(
        _dealias(vm.model),
        vm.adtype,
        vm.lp_function,
        vm.parameter_space,
        vm.vecinfo,
        vm.accs,
        vm.dimension,
        vm.lp_func,
        deepcopy(vm.adprep),
        copy(vm.sample),
    )
end

# Run the model once to collect everything needed to vectorise it: the latent values, plus
# the sampling sites that were hit exactly once (which provide the fast-path lookup).
function _prepare_vecmodel_info(
    model::Model,
    source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
)
    accs = AccumulatorTuple(VectorisedParamsAccumulator(), SitenameCountAccumulator())
    _, accs = evaluate(model, source, parameter_space, accs, EvalContext{false}())
    vecvals = vectorised_params(accs)
    sca = getacc(accs, Val{SITENAMECOUNT_ACCNAME}())

    # TODO(penelopeysm): This might be slower than necessary due to boxing of the
    # closed-over variables. It might be worth checking if a struct is better. However,
    # since this is only done once at VecModel construction time, it's not a huge deal.
    params = Union{}[]
    offset = 1
    varname_to_rat = VarNames.map_pairs!!(vecvals) do (vn, tval)
        value = get_internal_value(tval)
        value isa AbstractVector || throw(
            ArgumentError(
                "Expected a vector value for variable `$vn`, but got $(typeof(value)).",
            ),
        )
        range = offset:(offset+length(value)-1)
        params = vcat(params, value)
        offset = last(range) + 1
        RangeAndTransform(range, get_transform(tval))
    end
    vecinfo = VecInfo(sca, varname_to_rat)

    return vecinfo, params
end

"""
    lp(vecmodel::VecModel, params::AbstractVector)

Calculate the log-density of `vecmodel` at `params`.

!!! warning
    `params` is not copied, and the model may receive views into it. A model that writes
    into a latent variable (e.g. `x ~ MvNormal(...); x[1] += 1`) is therefore not
    supported: it may change `params` and give inconsistent results.
"""
lp(vecmodel::VecModel, params::AbstractVector) = vecmodel.lp_func(params)

"""
    lp_and_grad(vecmodel::VecModel, params::AbstractVector)

Returns a tuple (log_density, grad) where `log_density` is the log-density of `vecmodel` and
`grad` is its gradient at `params`.

!!! warning
    `params` is not copied, and the model may receive views into it. A model that writes
    into a latent variable (e.g. `x ~ MvNormal(...); x[1] += 1`) is therefore not
    supported: it may change `params` and give inconsistent results.
"""
function lp_and_grad(vecmodel::VecModel, params::AbstractVector)
    prepared_params = convert(typeof(vecmodel.sample), params)
    return if _use_closure(vecmodel.adtype)
        DI.value_and_gradient(
            vecmodel.lp_func,
            vecmodel.adprep,
            vecmodel.adtype,
            prepared_params,
        )
    else
        DI.value_and_gradient(
            _vecmodel_lp,
            vecmodel.adprep,
            vecmodel.adtype,
            prepared_params,
            DI.Constant(vecmodel.model.evaluation_fn),
            DI.Constant(vecmodel.model.args),
            DI.Constant(vecmodel.model.conditioned),
            DI.Constant(vecmodel.model.enzyme_copy_argnames),
            DI.Constant(_eval_context(vecmodel.adtype)),
            DI.Constant(vecmodel.lp_function),
            DI.Constant(vecmodel.parameter_space),
            DI.Constant(vecmodel.vecinfo),
            DI.Constant(vecmodel.accs),
        )
    end
end
function lp_and_grad(::VecModel{M,Nothing}, ::AbstractVector) where {M<:Model}
    throw(ArgumentError("This VecModel was constructed without an AD backend."))
end

"""
    lp_and_grad!(grad::AbstractVector, vecmodel::VecModel, params::AbstractVector)

Calculate the log-density and its gradient with respect to `params`, writing the gradient
into `grad`. Returns a tuple (log_density, grad).

!!! warning
    `params` is not copied, and the model may receive views into it. A model that writes
    into a latent variable (e.g. `x ~ MvNormal(...); x[1] += 1`) is therefore not
    supported: it may change `params` and give inconsistent results.
"""
function lp_and_grad!(grad::AbstractVector, vecmodel::VecModel, params::AbstractVector)
    prepared_params = convert(typeof(vecmodel.sample), params)
    return if _use_closure(vecmodel.adtype)
        DI.value_and_gradient!(
            vecmodel.lp_func,
            grad,
            vecmodel.adprep,
            vecmodel.adtype,
            prepared_params,
        )
    else
        DI.value_and_gradient!(
            _vecmodel_lp,
            grad,
            vecmodel.adprep,
            vecmodel.adtype,
            prepared_params,
            DI.Constant(vecmodel.model.evaluation_fn),
            DI.Constant(vecmodel.model.args),
            DI.Constant(vecmodel.model.conditioned),
            DI.Constant(vecmodel.model.enzyme_copy_argnames),
            DI.Constant(_eval_context(vecmodel.adtype)),
            DI.Constant(vecmodel.lp_function),
            DI.Constant(vecmodel.parameter_space),
            DI.Constant(vecmodel.vecinfo),
            DI.Constant(vecmodel.accs),
        )
    end
end
function lp_and_grad!(
    ::AbstractVector,
    ::VecModel{M,Nothing},
    ::AbstractVector,
) where {M<:Model}
    throw(ArgumentError("This VecModel was constructed without an AD backend."))
end

"""
    LogDensityProblems.capabilities(::Type{<:VecModel})

LogDensityProblems.LogDensityOrder{0} if the `VecModel` was constructed without an AD
backend, and LogDensityProblems.LogDensityOrder{1} otherwise.
"""
function LogDensityProblems.capabilities(
    ::Type{<:VecModel{M,A}},
) where {M<:Model,A<:Union{Nothing,NoAutoDiff}}
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(::Type{<:VecModel})
    return LogDensityProblems.LogDensityOrder{1}()
end
"""
    LogDensityProblems.dimension(vecmodel::VecModel)

The dimension of the parameter vector for `vecmodel`.
"""
LogDensityProblems.dimension(vecmodel::VecModel) = vecmodel.dimension
"""
    LogDensityProblems.logdensity(vecmodel::VecModel, params)

Evaluate the log-density of `vecmodel` at the given parameter vector. Equivalent to
[`lp`](@ref).
"""
function LogDensityProblems.logdensity(vecmodel::VecModel, params::AbstractVector{<:Number})
    return lp(vecmodel, params)
end
"""
    LogDensityProblems.logdensity_and_gradient(vecmodel::VecModel, params)

Evaluate the log-density of `vecmodel` at the given parameter vector and its gradient.
Equivalent to [`lp_and_grad`](@ref).
"""
function LogDensityProblems.logdensity_and_gradient(
    vecmodel::VecModel,
    params::AbstractVector{<:Number},
)
    return lp_and_grad(vecmodel, params)
end

"""
    Base.rand([rng::AbstractRNG,] vecmodel::VecModel) -> AbstractVector

Draw a parameter vector compatible with `vecmodel`.
"""
function Base.rand(rng::AbstractRNG, vecmodel::VecModel)
    return vectorise_latents(FromPrior(rng), vecmodel)
end
Base.rand(vecmodel::VecModel) = rand(default_rng(), vecmodel)

"""
    WrenCore.vectorise_latents(latent_source::AbstractLatentSource, vecmodel::VecModel)

Draw model parameters from `latent_source` and return them as a vector compatible with
`vecmodel`.
"""
function vectorise_latents(latent_source::AbstractLatentSource, vecmodel::VecModel)
    params = similar(vecmodel.sample)
    accs = AccumulatorTuple(VecSampleAccumulator(params, vecmodel.vecinfo))
    _, accs = evaluate(
        vecmodel.model,
        latent_source,
        vecmodel.parameter_space,
        accs,
        EvalContext{false}(),
    )
    return _get_vec_sample(accs)
end
