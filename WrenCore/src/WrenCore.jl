module WrenCore

using DensityInterface: DensityInterface, logdensityof
import DifferentiationInterface as DI
using DocStringExtensions: TYPEDFIELDS
using FillArrays: FillArrays
using LogDensityProblems: LogDensityProblems
using Plaice: Plaice
using Random: AbstractRNG, default_rng
using VarNames: VarNames, varname, VarName, VarNamedTuple, vnt_pretty_print
using StatsAPI: StatsAPI

function _issue_error(str)
    error(
        "$str.\n\nPlease report this issue at https://github.com/JuliaBayes/Wren.jl/issues. Thank you!\n",
    )
end

function _docs_link(str)
    return "https://juliabayes.org/Wren.jl/$str"
end

const WREN_COMPILATION_DEBUG = Ref(false)

# Parameter spaces
include("space.jl")

# Latent sources
include("latents/latents.jl")
include("latents/prior.jl")
include("latents/params.jl")

# Accumulators
include("accs/accs.jl")
include("accs/logp.jl")
include("accs/vnt.jl")
include("accs/params.jl")
include("accs/sitename_tracker.jl")
include("accs/accessors.jl")

# Model definition
include("model/expr.jl")
using .Expressions:
    AbstractWrenExpr,
    AddLogLikelihood,
    IID,
    Observe,
    OtherStatement,
    Batch,
    ReturnStatement,
    Sample,
    WrenExpr,
    _show_wren_exprs,
    _wren_expr_to_display_expr,
    get_top_level_symbol,
    to_wren_exprs

include("model/dataflow.jl")
include("model/model.jl")
include("model/submodel.jl")
include("model/emit.jl")

# Vectorisation
include("latents/vector.jl")
include("accs/vectorised_params.jl")
include("accs/vec_sample.jl")
include("vecmodel.jl")

# User-facing interface
include("interface.jl")
include("draw.jl")
include("test_ad.jl")
include("test_utils.jl")

export @wren,
    @addloglikelihood,
    @iid,
    @observe,
    @batch,
    AbstractLatentSource,
    AbstractTransform,
    AbstractParameterSpace,
    AbstractADCorrectnessTestSetting,
    AbstractAccumulator,
    AccumulatorTuple,
    LogPriorAccumulator,
    LogLikelihoodAccumulator,
    LogJacobianAccumulator,
    ParamsAccumulator,
    AsUnconstrainedVec,
    AsVec,
    ADIncorrectException,
    ADTestResult,
    BenchmarkResult,
    FromParams,
    FromPrior,
    FromVector,
    FixedTransform,
    UnconstrainedSpace,
    AsOriginal,
    TransformedValue,
    VecModel,
    DEFAULT_AD,
    VectorisedParamsAccumulator,
    vectorise_latents,
    WithBackend,
    WithExpectedResult,
    OriginalSpace,
    apply_parameter_space,
    condition,
    conditioned,
    decondition,
    iid,
    independent,
    rand_with_logp,
    evaluate,
    get_latent,
    vectorised_params,
    logjac,
    logjoint,
    logjoint_transformed,
    logprior,
    internal_values_as_vector,
    lp,
    lp_and_grad,
    lp_and_grad!,
    @bmodel,
    benchmark_model,
    test_ad,
    target_transform

# Reexports
using VarNames: @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras
export @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras

using DensityInterface: logdensityof
export logdensityof

using StatsAPI: loglikelihood, predict
export loglikelihood, predict

using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme
export AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme

using EnzymeCore: Forward, Reverse
export Forward, Reverse

using LinearAlgebra: I, Diagonal, Symmetric
export I, Diagonal, Symmetric

end # module WrenCore
