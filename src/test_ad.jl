using Chairmarks: Chairmarks, @be
using LinearAlgebra: norm
using Random: Xoshiro
using Statistics: median

"""
    AbstractADCorrectnessTestSetting

Different ways to check the correctness of an AD backend.
"""
abstract type AbstractADCorrectnessTestSetting end

"""
    WithBackend(adtype=AutoForwardDiff())

Check correctness against the log density and gradient produced by another AD backend.
"""
struct WithBackend{A<:DI.AbstractADType} <: AbstractADCorrectnessTestSetting
    adtype::A
end
WithBackend() = WithBackend(DI.AutoForwardDiff())

"""
    WithExpectedResult(lp, grad)

Check correctness against a known log density and gradient. Either can be left as `nothing`
to skip testing.
"""
struct WithExpectedResult{
    V<:Union{Number,Nothing},
    G<:Union{AbstractVector,Nothing}
} <: AbstractADCorrectnessTestSetting
    lp::V
    grad::G
end

"""
    NoTest()

Disable AD correctness checking in [`run_ad`](@ref).
"""
struct NoTest <: AbstractADCorrectnessTestSetting end

"""
    ADIncorrectException

Exception thrown when an AD backend returns an incorrect log-density or gradient.
"""
struct ADIncorrectException{VE,VA,GE,GA,T<:Real} <: Exception
    lp_expected::VE
    lp_actual::VA
    grad_expected::GE
    grad_actual::GA
    atol::T
    rtol::T
end
function Base.showerror(io::IO, exception::ADIncorrectException)
    lp_passed = isapprox(
        exception.lp_expected,
        exception.lp_actual;
        atol=exception.atol,
        rtol=exception.rtol,
    )
    grad_passed = isapprox(
        exception.grad_expected,
        exception.grad_actual;
        atol=exception.atol,
        rtol=exception.rtol,
    )
    failed = if !lp_passed && !grad_passed
        "lp and gradient"
    elseif !lp_passed
        "lp"
    else
        "gradient"
    end

    msg = """
    ADIncorrectException: incorrect $failed
      ├ atol     : $(exception.atol)
      ├ rtol     : $(exception.rtol)
      ├ expected : $((exception.lp_expected, exception.grad_expected))
      └ actual   : $((exception.lp_actual, exception.grad_actual))
    """
    print(io, msg)
    if !lp_passed
        min_atol = abs(exception.lp_expected - exception.lp_actual)
        min_rtol =
            min_atol / max(abs(exception.lp_expected), abs(exception.lp_actual))
        s = """
          lp check would have passed with:
               atol ≥ $min_atol"
            or rtol ≥ $min_rtol"
        """
        println(io)
        print(io, s)
    end
    if !grad_passed
        println(io)
        diff_norm = norm(exception.grad_expected - exception.grad_actual)
        max_norm = max(norm(exception.grad_expected), norm(exception.grad_actual))
        s = """
          Gradient check would have passed with:
               atol ≥ $diff_norm"
            or rtol ≥ $(diff_norm / max_norm)"
        """
        println(io)
        print(io, s)
    end
    return nothing
end

"""
    ModelRunResult

The correctness and benchmark results returned by [`run_ad`](@ref).

$(TYPEDFIELDS)
"""
struct ModelRunResult{
    M<:Model,
    V<:VarNamedTuple,
    VM<:Union{VecModel,Nothing},
    P<:Union{AbstractVector,Nothing},
    VE,
    GE,
    VA,
    GA,
    T<:Real,
}
    "The model that was tested."
    model::M
    "A VarNamedTuple sample drawn from the model's prior."
    vnt_params::V
    "The vectorised model that was tested."
    vecmodel::VM
    "The vector of parameters that was tested with `vecmodel`."
    params::P
    "The expected log-density value, or `nothing` if correctness was not checked."
    lp_expected::VE
    "The expected gradient of the log-density, or `nothing` if correctness was not checked."
    grad_expected::GE
    "The actual log-density value returned by the AD backend."
    lp_actual::VA
    "The actual gradient of the log-density returned by the AD backend."
    grad_actual::GA
    "The absolute tolerance used for correctness checking."
    atol::T
    "The relative tolerance used for correctness checking."
    rtol::T
    "Median benchmark for drawing a sample from the model, or `nothing` if not benchmarked."
    rand_bench::Union{Nothing,Chairmarks.Sample}
    "Median benchmark for evaluating the log-joint density, or `nothing` if not benchmarked."
    logjoint_bench::Union{Nothing,Chairmarks.Sample}
    "Median benchmark for evaluating the vectorised log-density, or `nothing` if not benchmarked."
    primal_bench::Union{Nothing,Chairmarks.Sample}
    "Median benchmark for evaluating the log-density and its gradient, or `nothing` if not benchmarked."
    grad_bench::Union{Nothing,Chairmarks.Sample}
end

_show_sample(s::Chairmarks.Sample) = sprint(show, MIME"text/plain"(), s)

function Base.show(io::IO, ::MIME"text/plain", result::ModelRunResult)
    lines = Pair{String,String}[]
    push!(lines, "model" => string(result.model.name))
    if result.vecmodel !== nothing
        push!(lines, "backend" => string(result.vecmodel.adtype))
    end
    if result.params !== nothing
        push!(lines, "n_params" => string(length(result.params)))
    end
    if result.rand_bench !== nothing
        push!(lines, "rand" => _show_sample(result.rand_bench))
    end
    if result.logjoint_bench !== nothing
        push!(lines, "logjoint" => _show_sample(result.logjoint_bench))
    end
    if result.primal_bench !== nothing
        push!(lines, "lp" => _show_sample(result.primal_bench))
    end
    if result.grad_bench !== nothing
        push!(lines, "lp_and_grad" => _show_sample(result.grad_bench))
        if result.primal_bench !== nothing
            ratio = result.grad_bench.time / result.primal_bench.time
            push!(lines, "grad / lp" => string(round(ratio; sigdigits=4)))
        end
    end
    if result.lp_expected !== nothing || result.grad_expected !== nothing
        push!(lines, "correctness" => "passed")
    end
    println(io, "ModelRunResult")
    for (i, (key, val)) in enumerate(lines)
        connector = i == length(lines) ? "└" : "├"
        padded_key = rpad(key, 12)
        if i == length(lines)
            print(io, "  $connector $padded_key: $val")
        else
            println(io, "  $connector $padded_key: $val")
        end
    end
    return nothing
end

"""
    benchmark_and_test(
        model::Wren.Model,
        adtype::Union{Nothing,DI.AbstractADType};
        kwargs...
    ) -> ModelRunResult

Test and benchmark a model, optionally including its vectorised form and AD.

Note that if `adtype` is specified, the corresponding backend package must be loaded first.

This function evaluates performance at multiple levels:

- **Model level**: benchmarks [`rand`](@ref) (sampling from the prior) and
  [`logjoint`](@ref) (log-joint density evaluation).
- **VecModel level**: benchmarks [`lp`](@ref) (log-density of the vectorised model) and
  [`lp_and_grad`](@ref) (log-density and its gradient).

If `test_correctness` is also specified, then the log-density and gradient returned by the
AD backend are compared against either a reference backend or a manually specified expected
result.

## Keyword arguments

The `lp_function` and `parameter_space` keywords are forwarded to [`VecModel`](@ref), and
have the same defaults.

"""
function benchmark_and_test(
    model::Model,
    adtype::Union{Nothing,DI.AbstractADType};
    benchmark_model::Bool=true,
    benchmark_primal::Bool=true,
    benchmark_ad::Bool=(adtype !== nothing),
    benchmark_seconds::Real=1.0,
    test_correctness::Union{AbstractADCorrectnessTestSetting,Bool}=NoTest(),
    atol::Real=100 * eps(),
    rtol::Real=sqrt(eps()),
    lp_function=logjoint_transformed,
    latent_source::AbstractLatentSource=FromPrior(),
    parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
    verbose::Bool=!isinteractive(),
)
    # Promote boolean test_ad to an AbstractADCorrectnessTestSetting
    test_correctness = test_correctness isa Bool ? (test_correctness ? WithBackend() : NoTest()) : test_correctness
    if test_correctness isa WithBackend && adtype == test_correctness.adtype
        @warn "Testing correctness of $adtype against itself — results will always pass. " *
              "Use a different reference backend or WithExpectedResult."
    end
    atol, rtol = promote(atol, rtol)

    _, accs =
        evaluate(model, latent_source, OriginalSpace(), Wren.default_accumulators())
    vnt_params = parameters(accs)

    # Model-level benchmarking: sample and logjoint
    rand_bench, logjoint_bench = if benchmark_model
        rng = Xoshiro(468) # avoid mutating global RNG
        rand(rng, model) # warmup
        rb = median(@be rand($rng, $model) seconds = benchmark_seconds)
        logjoint(model, vnt_params) # warmup
        ljb = median(@be logjoint($model, $vnt_params) seconds = benchmark_seconds)
        rb, ljb
    else
        nothing, nothing
    end
    if test_correctness === NoTest() && !benchmark_primal && !benchmark_ad
        # Nothing else to do, can return early.
        #! format: off
        return ModelRunResult(
            model, vnt_params, nothing, nothing, nothing, nothing, nothing,
            nothing, atol, rtol, rand_bench, logjoint_bench, nothing, nothing
        )
        #! format: on
    end

    # VecModel-level
    vecmodel = VecModel(
        model,
        adtype;
        params=FromParams(vnt_params, nothing),
        lp_function,
        parameter_space,
    )
    vec_params = vecmodel.sample

    lp_actual = test_correctness isa NoTest ? nothing : lp(vecmodel, vec_params)
    lp_expected = if test_correctness isa NoTest
        nothing
    elseif test_correctness isa WithExpectedResult
        test_correctness.lp
    else
        # Correctness is tested against no-backend. This usually doesn't really need to be
        # tested, except for Reactant where the computation graph is different.
        ref_vecmodel = VecModel(
            model,
            nothing;
            params=FromParams(vnt_params, nothing),
            lp_function,
            parameter_space,
        )
        lp(ref_vecmodel, vec_params)
    end
    primal_bench = if benchmark_primal
        lp(vecmodel, vec_params) # warm up
        median(@be lp($vecmodel, $vec_params) seconds = benchmark_seconds)
    else
        nothing
    end
    # If no AD testing / benchmarking, we can stop here
    if test_correctness isa NoTest && !benchmark_ad
        #! format: off
        return ModelRunResult(
            model, vnt_params, vecmodel, vec_params, lp_expected, nothing,
            lp_actual, nothing, atol, rtol, rand_bench, logjoint_bench,
            primal_bench, nothing
        )
        #! format: on
    end

    lp_actual, grad_actual = lp_and_grad(vecmodel, vec_params)
    # standardise the array type returned. DI is agnostic towards this.
    # https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/754
    grad_actual = collect(grad_actual)

    grad_expected = if test_correctness isa NoTest
        nothing
    elseif test_correctness isa WithExpectedResult
        test_correctness.grad === nothing ? nothing : collect(test_correctness.grad)
    else
        reference = VecModel(
            vecmodel.model,
            test_correctness.adtype,
            vecmodel.lp_function,
            vecmodel.parameter_space,
            vecmodel.ranges_and_transforms,
            vecmodel.sample,
        )
        _, grad = lp_and_grad(reference, vec_params)
        collect(grad)
    end

    if !(test_correctness isa NoTest)
        lp_passed =
            lp_expected !== nothing && isapprox(lp_expected, lp_actual; atol, rtol)
        grad_passed =
            grad_expected !== nothing && isapprox(grad_expected, grad_actual; atol, rtol)
        if !lp_passed || !grad_passed
            throw(
                ADIncorrectException(
                    lp_expected,
                    lp_actual,
                    grad_expected,
                    grad_actual,
                    atol,
                    rtol,
                ),
            )
        end
    end

    grad_bench = if benchmark_model
        lp_and_grad(vecmodel, vec_params)
        median(@be lp_and_grad($vecmodel, $vec_params) seconds = benchmark_seconds)
    else
        nothing
    end

    result = ModelRunResult(
        model,
        vnt_params,
        vecmodel,
        vec_params,
        lp_expected,
        grad_expected,
        lp_actual,
        grad_actual,
        atol,
        rtol,
        rand_bench,
        logjoint_bench,
        primal_bench,
        grad_bench,
    )

    if verbose
        show(stdout, MIME"text/plain"(), result)
        println()
    end

    return result
end

"""
    test_ad(model, adtype; kwargs...) -> ModelRunResult

Check the correctness of `adtype` on `model` without benchmarking.

Equivalent to `benchmark_and_test(model, adtype; benchmark_model=false, benchmark_ad=false, kwargs...)`.
"""
function test_ad(
    model::Model,
    adtype::DI.AbstractADType;
    test_correctness::Union{AbstractADCorrectnessTestSetting,Bool}=WithBackend(),
    benchmark_model=false,
    benchmark_primal=false,
    benchmark_ad=false,
    kwargs...,
)
    return benchmark_and_test(model, adtype; test_correctness, benchmark_model, benchmark_primal, benchmark_ad, kwargs...)
end

"""
    benchmark_ad(model, adtype; kwargs...) -> ModelRunResult

Benchmark `adtype` on `model` without checking correctness.

Equivalent to `benchmark_and_test(model, adtype; test_ad=NoTest(), kwargs...)`.
"""
function benchmark_ad(
    model::Model, adtype::DI.AbstractADType; test_correctness=NoTest(), kwargs...
)
    return benchmark_and_test(model, adtype; test_correctness, kwargs...)
end

"""
    @bmodel model
    @bmodel model adtype

Benchmark a model. The one-argument form benchmarks `rand`, `logjoint`, and `lp` only. The
two-argument form also benchmarks `lp_and_grad` with the given AD backend.
"""
macro bmodel(model)
    return :(benchmark_and_test($(esc(model)), nothing; benchmark_ad=false))
end
macro bmodel(model, adtype)
    return :(benchmark_and_test($(esc(model)), $(esc(adtype))))
end
