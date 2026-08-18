"""
    VecModel(model, adtype; params=FromPrior(),
             getlogp=getlogjoint_transformed,
             parameter_space=UnconstrainedSpace())

A model represented by a single parameter vector.

`params` may be an existing `VarNamedTuple` of vector values, or an
[`AbstractLatentSource`](@ref) used to evaluate the model and generate one. `VecModel`
assumes that the model's latent variables, their dimensions, and their order remain fixed
between evaluations.

Use [`rand`](@ref) to draw a compatible parameter vector, [`lp`](@ref) to evaluate its
transformed-space log joint density, and [`lp_and_grad`](@ref) or [`lp_and_grad!`](@ref) to
differentiate it with the supplied `adtype`. Pass `nothing` as the `adtype` to construct a
log-density-only model without loading or preparing an AD backend.

By default the vector is in [`UnconstrainedSpace`](@ref) and `lp` returns
[`getlogjoint_transformed`](@ref). Use the `parameter_space` and `getlogp` keywords to
select different behaviour.
"""
struct _VecModelLP{
    E,
    A<:NamedTuple,
    C<:VarNamedTuple,
    F,
    S<:AbstractParameterSpace,
    R<:VarNamedTuple,
    AC<:AccumulatorTuple,
}
    evaluation_fn::E
    model_args::A
    conditioned::C
    getlogp::F
    parameter_space::S
    ranges_and_transforms::R
    accs::AC
end

function _vecmodel_lp(
    params::AbstractVector,
    evaluation_fn,
    model_args::NamedTuple,
    conditioned::VarNamedTuple,
    getlogp,
    parameter_space::AbstractParameterSpace,
    ranges_and_transforms::VarNamedTuple,
    accs::AccumulatorTuple,
)
    source = FromVector(params, ranges_and_transforms)
    _, accs = evaluation_fn(source, parameter_space, accs, model_args, conditioned)
    return getlogp(accs)
end

function (f::_VecModelLP)(params::AbstractVector)
    return _vecmodel_lp(
        params,
        f.evaluation_fn,
        f.model_args,
        f.conditioned,
        f.getlogp,
        f.parameter_space,
        f.ranges_and_transforms,
        f.accs,
    )
end

_use_closure(::DI.AutoReverseDiff{compile}) where {compile} = compile
_use_closure(::DI.AbstractADType) = false

struct VecModel{
    M<:Model,
    A<:Union{DI.AbstractADType,Nothing},
    F,
    S<:AbstractParameterSpace,
    R<:VarNamedTuple,
    AC<:AccumulatorTuple,
    L<:_VecModelLP,
    P,
    X<:AbstractVector,
}
    model::M
    adtype::A
    getlogp::F
    parameter_space::S
    ranges_and_transforms::R
    accs::AC
    dimension::Int
    lp_func::L
    adprep::P
    sample::X
end

# Unlike in DynamicPPL, this has to be an outer constructor because the ReactantExt requires
# access to the direct (generated) constructor.
function VecModel(
    model::M,
    adtype::A,
    getlogp::F,
    parameter_space::S,
    ranges_and_transforms::R,
    params::X,
) where {
    M<:Model,
    A<:Union{DI.AbstractADType,Nothing},
    F,
    S<:AbstractParameterSpace,
    R<:VarNamedTuple,
    X<:AbstractVector,
}
    accs = AccumulatorTuple(
        LogPriorAccumulator(),
        LogJacobianAccumulator(),
        LogLikelihoodAccumulator(),
    )
    lp_func = _VecModelLP(
        model.evaluation_fn,
        model.args,
        model.conditioned,
        getlogp,
        parameter_space,
        ranges_and_transforms,
        accs,
    )
    args = (
        model.evaluation_fn,
        model.args,
        model.conditioned,
        getlogp,
        parameter_space,
        ranges_and_transforms,
        accs,
    )
    adprep = if adtype === nothing
        nothing
    elseif _use_closure(adtype)
        DI.prepare_gradient(lp_func, adtype, params)
    else
        DI.prepare_gradient(_vecmodel_lp, adtype, params, map(DI.Constant, args)...)
    end
    return VecModel(
        model,
        adtype,
        getlogp,
        parameter_space,
        ranges_and_transforms,
        accs,
        length(params),
        lp_func,
        adprep,
        params,
    )
end

function VecModel(
    model::Model,
    adtype::Union{DI.AbstractADType,Nothing};
    params::Union{VarNamedTuple,AbstractLatentSource}=FromPrior(),
    getlogp=getlogjoint_transformed,
    parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
)
    values = if params isa VarNamedTuple
        params
    else
        _draw_vector_values(model, params, parameter_space)
    end
    _validate_vector_values(values, parameter_space)
    ranges_and_transforms, sample = _ranges_and_params(values)
    return VecModel(model, adtype, getlogp, parameter_space, ranges_and_transforms, sample)
end

function _validate_vector_values(
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

function _draw_vector_values(
    model::Model,
    source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
)
    accs = AccumulatorTuple(VectorValueAccumulator())
    _, accs = evaluate(model, source, parameter_space, accs)
    return get_vector_values(accs)
end

function _ranges_and_params(values::VarNamedTuple)
    ranges_and_transforms = VarNamedTuple()
    params = Union{}[]
    offset = 1

    for (vn, tval) in pairs(values)
        value = get_internal_value(tval)
        value isa AbstractVector || throw(
            ArgumentError(
                "Expected a vector value for variable `$vn`, but got $(typeof(value)).",
            ),
        )

        range = offset:(offset+length(value)-1)
        template = get(values.data, VarNames.getsym(vn), VarNames.NoTemplate())
        ranges_and_transforms = VarNames.templated_setindex!!(
            ranges_and_transforms,
            RangeAndTransform(range, get_transform(tval)),
            vn,
            template,
        )
        params = vcat(params, value)
        offset = last(range) + 1
    end

    return ranges_and_transforms, params
end

"""
    lp(vecmodel::VecModel, params::AbstractVector)

Evaluate `vecmodel.getlogp` in the vector model's parameter space.
"""
function lp(vecmodel::VecModel, params::AbstractVector)
    return vecmodel.lp_func(params)
end

"""
    lp_and_grad(vecmodel::VecModel, params::AbstractVector)

Return the log density and its gradient with respect to `params`.

Throws an `ArgumentError` if `vecmodel` was constructed with `nothing` instead of an AD
backend.
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
            DI.Constant(vecmodel.getlogp),
            DI.Constant(vecmodel.parameter_space),
            DI.Constant(vecmodel.ranges_and_transforms),
            DI.Constant(vecmodel.accs),
        )
    end
end

function lp_and_grad(vecmodel::VecModel{M,Nothing}, params::AbstractVector) where {M<:Model}
    throw(ArgumentError("This VecModel was constructed without an AD backend."))
end

"""
    lp_and_grad!(grad::AbstractVector, vecmodel::VecModel, params::AbstractVector)

Return the log density and its gradient with respect to `params`, writing the gradient into
`grad`.

Throws an `ArgumentError` if `vecmodel` was constructed with `nothing` instead of an AD
backend.
"""
function lp_and_grad!(grad::AbstractVector, vecmodel::VecModel, params::AbstractVector)
    length(grad) == vecmodel.dimension || throw(
        DimensionMismatch(
            "VecModel expects a gradient buffer of length $(vecmodel.dimension), but " *
            "received one of length $(length(grad)).",
        ),
    )
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
            DI.Constant(vecmodel.getlogp),
            DI.Constant(vecmodel.parameter_space),
            DI.Constant(vecmodel.ranges_and_transforms),
            DI.Constant(vecmodel.accs),
        )
    end
end

function lp_and_grad!(
    grad::AbstractVector,
    vecmodel::VecModel{M,Nothing},
    params::AbstractVector,
) where {M<:Model}
    throw(ArgumentError("This VecModel was constructed without an AD backend."))
end

function LogDensityProblems.capabilities(::Type{<:VecModel{M,Nothing}}) where {M<:Model}
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.capabilities(::Type{<:VecModel})
    return LogDensityProblems.LogDensityOrder{1}()
end

LogDensityProblems.dimension(vecmodel::VecModel) = vecmodel.dimension

function LogDensityProblems.logdensity(vecmodel::VecModel, params::AbstractVector{<:Real})
    return lp(vecmodel, params)
end

function LogDensityProblems.logdensity_and_gradient(
    vecmodel::VecModel,
    params::AbstractVector{<:Real},
)
    return lp_and_grad(vecmodel, params)
end

"""Draw a parameter vector compatible with `vecmodel`."""
function Base.rand(rng::AbstractRNG, vecmodel::VecModel)
    values = _draw_vector_values(vecmodel.model, FromPrior(rng), vecmodel.parameter_space)
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
