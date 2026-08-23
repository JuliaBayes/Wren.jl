using ADTypes: NoAutoDiff # DI doesn't export

const AUTO_RUNTIME_ACTIVITY = Ref(true)

"""
    Wren.set_auto_runtime_activity!(val::Bool)

Set Wren's automatic runtime activity behaviour.

When using Enzyme as the AD backend, depending on how the model is constructed, it may be
necessary to enable runtime activity. (Wren-specific docs will follow soon, but for more
information please see the [Enzyme
docs](https://enzyme.mit.edu/julia/stable/faq/#faq-runtime-activity).) This can often allow
more expressive models to be differentiated, but comes at a performance cost.

Wren recommends using reverse-mode Enzyme, i.e., `AutoEnzyme(; mode=Reverse)` as the AD
backend, which does not enable runtime activity. **By default, Wren will automatically
enable runtime activity if Enzyme requires it.**

This behaviour can be disabled by calling

```julia
Wren.set_auto_runtime_activity!(false)`

This will cause Wren to throw an error instead if runtime activity is required.
"""
function set_auto_runtime_activity!(val::Bool)
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
    C<:VarNamedTuple,
    F,
    S<:AbstractParameterSpace,
    R<:RangesAndTransforms,
    AC<:AccumulatorTuple,
}
    evaluation_fn::E
    model_args::A
    conditioned::C
    lp_function::F
    parameter_space::S
    ranges_and_transforms::R
    accs::AC
end
function (f::_VecModelLP)(params::AbstractVector)
    return _vecmodel_lp(
        params,
        f.evaluation_fn,
        f.model_args,
        f.conditioned,
        f.lp_function,
        f.parameter_space,
        f.ranges_and_transforms,
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
    conditioned::VarNamedTuple,
    lp_function,
    parameter_space::AbstractParameterSpace,
    ranges_and_transforms::RangesAndTransforms,
    accs::AccumulatorTuple,
)
    source = FromVector(params, ranges_and_transforms)
    _, accs = evaluation_fn(source, parameter_space, accs, model_args, conditioned)
    return lp_function(accs)
end

"""
    VecModel(
        model::Model,
        adtype::Union{ADTypes.AbstractADType,Nothing};
        lp_function=logjoint_transformed,
        latent_source=FromPrior(),
        parameter_space=UnconstrainedSpace()
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
  `VarNamedTuple` of values, wrap it in [`FromParams`](@ref).

  If you have already run the model once, you can avoid reevaluation entirely by calling the
  positional constructor with a `RangesAndTransforms` and a parameter vector. This is mostly
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
"""
struct VecModel{
    M<:Model,
    A<:Union{DI.AbstractADType,Nothing},
    F,
    S<:AbstractParameterSpace,
    R<:RangesAndTransforms,
    AC<:AccumulatorTuple,
    L<:_VecModelLP,
    P,
    X<:AbstractVector,
}
    model::M
    adtype::A
    lp_function::F
    parameter_space::S
    ranges_and_transforms::R
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
    ranges_and_transforms::R,
    params::X,
) where {
    M<:Model,
    A<:Union{DI.AbstractADType,Nothing},
    F,
    S<:AbstractParameterSpace,
    R<:RangesAndTransforms,
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
        lp_function,
        parameter_space,
        ranges_and_transforms,
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
            DI.Constant(lp_function),
            DI.Constant(parameter_space),
            DI.Constant(ranges_and_transforms),
            DI.Constant(accs),
        )
    end
    vm = VecModel(
        model,
        adtype,
        lp_function,
        parameter_space,
        ranges_and_transforms,
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
)
    # Get vectorised parameters.
    values, sitename_tracker = _prepare_vecmodel_info(model, latent_source, parameter_space)
    # Check that they're sensible, and flatten them into a single vector.
    _validate_vectorised_params(values, parameter_space)
    ranges_and_transforms, sample = _ranges_and_params(values, sitename_tracker)
    return VecModel(
        model,
        adtype,
        lp_function,
        parameter_space,
        ranges_and_transforms,
        sample,
    )
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

# overloaded in EnzymeExt as it depends on runtime activity error which isn't in Core
_handle_enzyme(@nospecialize(vm::VecModel)) = vm

function _validate_vectorised_params(
    values::VarNamedTuple,
    parameter_space::AbstractParameterSpace,
)
    for (vn, tval) in pairs(values)
        expected = target_transform(parameter_space, vn)
        get_transform(tval) == expected || throw(
            ArgumentError(
                "The value for variable `$vn` uses $(get_transform(tval)), but " *
                "the selected parameter space requires $expected.",
            ),
        )
    end
    return nothing
end

# Run the model once to collect everything needed to vectorise it: the latent values, plus
# the sampling sites that were hit exactly once (which provide the fast-path lookup).
function _prepare_vecmodel_info(
    model::Model,
    source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
)
    accs = AccumulatorTuple(VectorisedParamsAccumulator(), SiteNameTracker())
    _, accs = evaluate(model, source, parameter_space, accs)
    return vectorised_params(accs), accs[SITENAME_TRACKER_ACCNAME]
end
function _ranges_and_params(
    values::VarNamedTuple,
    sitename_tracker::SiteNameTracker=SiteNameTracker(),
)
    params = Union{}[]
    offset = 1
    # `map_pairs!!` rewrites the leaves in place of `values`, which keeps the structure of
    # the VarNamedTuple intact. Rebuilding it leaf-by-leaf instead would silently drop any
    # submodel that sampled nothing, since those are empty sub-VarNamedTuples with no leaves
    # to iterate over -- and `scope(::FromVector, vn)` needs every submodel prefix to be
    # present. It also means the ranges we hand out and the order in which we concatenate
    # the values come from a single traversal, so they cannot disagree.
    slow_vnt = VarNames.map_pairs!!(values) do (vn, tval)
        value = get_internal_value(tval)
        value isa AbstractVector || throw(
            ArgumentError(
                "Expected a vector value for variable `$vn`, but got $(typeof(value)).",
            ),
        )
        range = offset:(offset+length(value)-1)
        params = vcat(params, value)
        offset = last(range) + 1
        return RangeAndTransform(range, get_transform(tval))
    end
    return RangesAndTransforms(sitename_tracker, slow_vnt), params
end

"""
    lp(vecmodel::VecModel, params::AbstractVector)

Calculate the log-density of `vecmodel` at `params`.
"""
lp(vecmodel::VecModel, params::AbstractVector) = vecmodel.lp_func(params)

"""
    lp_and_grad(vecmodel::VecModel, params::AbstractVector)

Returns a tuple (log_density, grad) where `log_density` is the log-density of `vecmodel` and
`grad` is its gradient at `params`.
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
            DI.Constant(vecmodel.lp_function),
            DI.Constant(vecmodel.parameter_space),
            DI.Constant(vecmodel.ranges_and_transforms),
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
            DI.Constant(vecmodel.lp_function),
            DI.Constant(vecmodel.parameter_space),
            DI.Constant(vecmodel.ranges_and_transforms),
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
    # TODO: this is inefficient -- should write directly into the vector
    values, _ =
        _prepare_vecmodel_info(vecmodel.model, FromPrior(rng), vecmodel.parameter_space)
    ranges_and_transforms, params = _ranges_and_params(values)
    ranges_and_transforms == vecmodel.ranges_and_transforms || throw(
        ArgumentError(
            "The model produced a parameter layout that is incompatible with this " *
            "VecModel.",
        ),
    )
    return params
end
Base.rand(vecmodel::VecModel) = rand(default_rng(), vecmodel)
