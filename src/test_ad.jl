using Chairmarks: @be
using Statistics: median

"""Different ways to check the correctness of an AD backend."""
abstract type AbstractADCorrectnessTestSetting end

"""
    WithBackend(adtype=AutoForwardDiff())

Check correctness against the value and gradient produced by another AD backend.
"""
struct WithBackend{A<:DI.AbstractADType} <: AbstractADCorrectnessTestSetting
    adtype::A
end
WithBackend() = WithBackend(DI.AutoForwardDiff())

"""
    WithExpectedResult(value, grad)

Check correctness against a known log density and gradient.
"""
struct WithExpectedResult{V<:Real,G<:AbstractVector} <:
       AbstractADCorrectnessTestSetting
    value::V
    grad::G
end

"""Disable AD correctness checking in [`run_ad`](@ref)."""
struct NoTest <: AbstractADCorrectnessTestSetting end

"""Exception thrown when an AD backend returns an incorrect value or gradient."""
struct ADIncorrectException{VE,VA,GE,GA,T<:Real} <: Exception
    value_expected::VE
    value_actual::VA
    grad_expected::GE
    grad_actual::GA
    atol::T
    rtol::T
end

function Base.showerror(io::IO, exception::ADIncorrectException)
    value_passed = isapprox(
        exception.value_expected,
        exception.value_actual;
        atol=exception.atol,
        rtol=exception.rtol,
    )
    grad_passed = isapprox(
        exception.grad_expected,
        exception.grad_actual;
        atol=exception.atol,
        rtol=exception.rtol,
    )
    failed = if !value_passed && !grad_passed
        "value and gradient"
    elseif !value_passed
        "value"
    else
        "gradient"
    end

    println(io, "ADIncorrectException: incorrect $failed")
    println(io, "  ├ atol     : $(exception.atol)")
    println(io, "  ├ rtol     : $(exception.rtol)")
    println(io, "  ├ expected : $((exception.value_expected, exception.grad_expected))")
    print(io, "  └ actual   : $((exception.value_actual, exception.grad_actual))")
    return nothing
end

"""The correctness and benchmark results returned by [`run_ad`](@ref)."""
struct ADResult{VM<:VecModel,P<:AbstractVector,VE,GE,VA,GA,T<:Real}
    vecmodel::VM
    params::P
    value_expected::VE
    grad_expected::GE
    value_actual::VA
    grad_actual::GA
    atol::T
    rtol::T
    primal_time::Union{Nothing,Float64}
    grad_time::Union{Nothing,Float64}
end

function _pretty_time(seconds::Real)
    return if seconds < 1e-6
        "$(round(seconds * 1e9; sigdigits=4)) ns"
    elseif seconds < 1e-3
        "$(round(seconds * 1e6; sigdigits=4)) μs"
    elseif seconds < 1
        "$(round(seconds * 1e3; sigdigits=4)) ms"
    else
        "$(round(seconds; sigdigits=4)) s"
    end
end

function Base.show(io::IO, ::MIME"text/plain", result::ADResult)
    checked = result.value_expected === nothing ? "not checked" : "passed"
    println(io, "ADResult")
    println(io, "  ├ model       : $(result.vecmodel.model.name)")
    println(io, "  ├ backend     : $(result.vecmodel.adtype)")
    println(io, "  ├ correctness : $checked")
    println(io, "  ├ value       : $(result.value_actual)")
    println(io, "  ├ gradient    : $(result.grad_actual)")
    if result.primal_time !== nothing
        println(io, "  ├ lp          : $(_pretty_time(result.primal_time))")
        println(io, "  ├ lp + grad   : $(_pretty_time(result.grad_time))")
        println(
            io,
            "  ├ grad / lp   : $(round(result.grad_time / result.primal_time; sigdigits=4))",
        )
    end
    print(io, "  └ params      : $(result.params)")
    return nothing
end

"""
    run_ad(model, adtype; kwargs...) -> ADResult

Test and optionally benchmark `adtype` on a vectorised model.

By default, correctness is checked against ForwardDiff using [`WithBackend`](@ref), and
benchmarking is disabled. Pass `test=NoTest()` (or `false`) to skip correctness checking,
`test=WithExpectedResult(value, grad)` to supply a known answer, or `benchmark=true` to
benchmark prepared calls to [`lp`](@ref) and [`lp_and_grad`](@ref).

The `getlogp` and `parameter_space` keywords are forwarded to [`VecModel`](@ref), and have
the same defaults. The requested AD backend package, and any reference backend package,
must already be loaded.
"""
function run_ad(
    model::Model,
    adtype::DI.AbstractADType;
    test::Union{AbstractADCorrectnessTestSetting,Bool}=WithBackend(),
    benchmark::Bool=false,
    benchmark_seconds::Real=1.0,
    atol::Real=100 * eps(),
    rtol::Real=sqrt(eps()),
    getlogp=getlogjoint_transformed,
    rng::AbstractRNG=default_rng(),
    parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
    params::Union{Nothing,AbstractVector}=nothing,
    verbose::Bool=true,
)
    setting = test isa Bool ? (test ? WithBackend() : NoTest()) : test
    atol, rtol = promote(atol, rtol)
    vecmodel = VecModel(
        model,
        adtype;
        params=FromPrior(rng),
        getlogp,
        parameter_space,
    )
    params = params === nothing ? rand(rng, vecmodel) : collect(params)

    value_actual, grad_actual = lp_and_grad(vecmodel, params)
    grad_actual = collect(grad_actual)

    value_expected, grad_expected = if setting isa NoTest
        nothing, nothing
    elseif setting isa WithExpectedResult
        setting.value, collect(setting.grad)
    else
        reference = VecModel(
            vecmodel.model,
            setting.adtype,
            vecmodel.getlogp,
            vecmodel.parameter_space,
            vecmodel.ranges_and_transforms,
            vecmodel._sample,
        )
        value, grad = lp_and_grad(reference, params)
        value, collect(grad)
    end

    if !(setting isa NoTest)
        value_passed = isapprox(value_expected, value_actual; atol, rtol)
        grad_passed = isapprox(grad_expected, grad_actual; atol, rtol)
        if !value_passed || !grad_passed
            throw(
                ADIncorrectException(
                    value_expected,
                    value_actual,
                    grad_expected,
                    grad_actual,
                    atol,
                    rtol,
                ),
            )
        end
    end

    primal_time, grad_time = if benchmark
        lp(vecmodel, params)
        primal_benchmark = @be lp($vecmodel, $params) seconds=benchmark_seconds
        lp_and_grad(vecmodel, params)
        grad_benchmark = @be lp_and_grad($vecmodel, $params) seconds=benchmark_seconds
        median(primal_benchmark).time, median(grad_benchmark).time
    else
        nothing, nothing
    end

    result = ADResult(
        vecmodel,
        params,
        value_expected,
        grad_expected,
        value_actual,
        grad_actual,
        atol,
        rtol,
        primal_time,
        grad_time,
    )
    if verbose
        show(stdout, MIME"text/plain"(), result)
        println()
    end
    return result
end
