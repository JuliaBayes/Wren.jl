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

function _docs_link(str)
    return "https://juliabayes.org/Wren.jl/$str"
end

const WREN_COMPILATION_DEBUG = Ref(false)

# `file:line` for a macro call's `__source__`. Files inside WrenCore are shown relative to
# the package, e.g. `WrenCore/src/model/dataflow.jl:10`, rather than as full paths.
function _source_location(source::LineNumberNode)
    source.file === nothing && return "unknown location"
    file = String(source.file)
    pkgdir = dirname(@__DIR__)
    if startswith(file, pkgdir)
        file = joinpath("WrenCore", relpath(file, pkgdir))
    end
    return string(replace(file, '\\' => '/'), ":", source.line)
end

# Callable structs have no name, so show their type's name instead.
_function_name(f::Function) = string(nameof(f))
_function_name(f) = string(nameof(typeof(f)))

# An expression which evaluates to a description of where a macro was called from, for use
# in error and debug messages. On Julia 1.13+, this includes the enclosing function, so the
# macro must be called inside a function.
function _caller_location(source::LineNumberNode)
    location = _source_location(source)
    @static if isdefined(Base, Symbol("@__FUNCTION__"))
        fn = Expr(:macrocall, GlobalRef(Base, Symbol("@__FUNCTION__")), source)
        return :($(Base.string)($_function_name($fn), " at ", $location))
    else
        return location
    end
end

"""
    @_issue_error msg

Throw an error saying that `msg` is a bug in Wren, asking the user to report it.

The message says where `@_issue_error` was called from (on Julia 1.13+, including the
enclosing function), so it must be called inside a function. This is a macro rather than a
function so that the location is the caller's.
"""
macro _issue_error(msg)
    return :($_throw_issue_error($(esc(msg)), $(_caller_location(__source__))))
end

@noinline function _throw_issue_error(msg, location)
    error(
        "$msg.\n\nThis is an internal error in Wren (from $location). Please report this " *
        "issue at https://github.com/JuliaBayes/Wren.jl/issues. Thank you!\n",
    )
end

"""
    @_compilation_debug msg [key=value | value]...

If `WrenCore.WREN_COMPILATION_DEBUG[]` is `true`, log `msg` and the given key-value pairs
at `Info` level, as `@info` would. The message is prefixed with where
`@_compilation_debug` was called from (on Julia 1.13+, including the enclosing function),
so it must be called inside a function. Does nothing otherwise.
"""
macro _compilation_debug(msg, kwargs...)
    file = __source__.file === nothing ? "unknown" : String(__source__.file)
    message = :($(Base.string)("[", $(_caller_location(__source__)), "] ", $msg))
    log = Expr(
        :macrocall,
        GlobalRef(Base, Symbol("@info")),
        __source__,
        message,
        kwargs...,
        :(_file = $file),
        :(_line = $(__source__.line)),
    )
    flag = GlobalRef(@__MODULE__, :WREN_COMPILATION_DEBUG)
    # Escape everything so that `@info` sees the key-value pairs as written.
    return esc(:(
        if $flag[]
            $log
        end
    ))
end

# Parameter spaces
include("space.jl")

# Latent sources
include("latents/latents.jl")
include("latents/void.jl")
include("latents/prior.jl")
include("latents/values.jl")
include("latents/interval.jl")

# Accumulators
include("accs/accs.jl")
include("accs/logp.jl")
include("accs/vnt.jl")
include("accs/params.jl")
include("accs/sitename_tracker.jl")
include("accs/tq.jl")
include("accs/accessors.jl")

# Model definition
include("model/expr.jl")
using .Expressions:
    AbstractWrenExpr,
    AddLogLikelihood,
    TrackedQuantity,
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
include("evaluate.jl")
include("condition.jl")
include("test_ad.jl")
include("test_utils.jl")
include("optimization.jl")

# Overloaded in extensions
function to_flexichain end
function posteriordb_model end
function available_posteriordb_models end

export @wren,
    @addloglikelihood,
    @iid,
    @sample,
    @observe,
    @batch,
    @track,
    Model,
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
    TrackedQuantityAccumulator,
    AsUnconstrainedVec,
    AsVec,
    ADIncorrectException,
    ADTestResult,
    BenchmarkResult,
    FromValues,
    FromPrior,
    FromUnconstrainedInterval,
    FromVector,
    UnconstrainedSpace,
    AsOriginal,
    TransformedValue,
    VecModel,
    DEFAULT_AD,
    VectorisedParamsAccumulator,
    vector_parameters,
    WithBackend,
    WithExpectedResult,
    OriginalSpace,
    apply_parameter_space,
    condition,
    conditioned,
    decondition,
    iid,
    independent,
    simulate,
    ModelResult,
    return_value,
    tracked_quantities,
    TrackedQuantitySpec,
    AllQuantities,
    SomeQuantities,
    NoQuantities,
    evaluate,
    EvalContext,
    get_latent,
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
    reduce_copies,
    target_transform,
    # Optimisation
    optimize,
    MAP,
    MLE,
    # FlexiChains
    to_flexichain,
    # PosteriorDB
    posteriordb_model,
    available_posteriordb_models

# Reexports
using VarNames: @vn, @vnt, VarNamedTuple, VarName, parameters
export @vn, @vnt, VarNamedTuple, VarName, parameters

using DensityInterface: logdensityof
export logdensityof

using StatsAPI: loglikelihood
export loglikelihood

using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme
export AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme

using EnzymeCore: Forward, Reverse
export Forward, Reverse

using LinearAlgebra: I, Diagonal, Symmetric
export I, Diagonal, Symmetric

end # module WrenCore
