# if VERSION < v"1.13-"
#     error("This script requires Julia 1.13 or later")
# end

using ADTypes
using Chairmarks: @be
# using Differ
using Distributions
using DynamicPPL
using Enzyme
using ForwardDiff
using LogDensityProblems
using Mooncake
using Printf
using Random: Xoshiro
using Reactant
using ReverseDiff
using Statistics: median
using WrenCore

include("pmtransforms.jl")
include("models.jl")
# include("differ_rules.jl")

using .BenchmarkModels

const BENCHMARK_SECONDS = parse(Float64, get(ENV, "WREN_BENCH_SECONDS", "1.0"))

const BACKENDS = (
    (name="ForwardDiff", wren=ADTypes.AutoForwardDiff(), dppl=ADTypes.AutoForwardDiff()),
    (name="ReverseDiff", wren=ADTypes.AutoReverseDiff(), dppl=ADTypes.AutoReverseDiff()),
    (
        name="ReverseDiffCpl",
        wren=ADTypes.AutoReverseDiff(; compile=true),
        dppl=ADTypes.AutoReverseDiff(; compile=true),
    ),
    (
        name="MooncakeFwd",
        wren=ADTypes.AutoMooncakeForward(),
        dppl=ADTypes.AutoMooncakeForward(),
    ),
    (name="MooncakeRvs", wren=ADTypes.AutoMooncake(), dppl=ADTypes.AutoMooncake()),
    # (name="DifferFwd", wren=Differ.AutoDifferForwards(), dppl=Differ.AutoDifferForwards()),
    # (name="DifferRvs", wren=Differ.AutoDifferReverse(), dppl=Differ.AutoDifferReverse()),
    (
        name="EnzymeFwd",
        wren=ADTypes.AutoEnzyme(; mode=Forward),
        dppl=ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward)),
    ),
    (
        name="EnzymeRvs",
        wren=ADTypes.AutoEnzyme(; mode=Reverse),
        dppl=ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)),
    ),
    (
        name="ReactantRvs",
        wren=ADTypes.AutoReactant(ADTypes.AutoEnzyme(; mode=Reverse)),
        dppl=nothing,
    ),
)

function dppl_density(model, adtype)
    return DynamicPPL.LogDensityFunction(
        model,
        DynamicPPL.getlogjoint_internal,
        DynamicPPL.LinkAll();
        adtype,
    )
end

function pretty_time(seconds)
    return if seconds < 1e-6
        @sprintf("%.3f ns", seconds * 1e9)
    elseif seconds < 1e-3
        @sprintf("%.3f μs", seconds * 1e6)
    elseif seconds < 1
        @sprintf("%.3f ms", seconds * 1e3)
    else
        @sprintf("%.3f s", seconds)
    end
end

function benchmark_pair(models, backend, seed)
    wren_model = WrenCore.VecModel(
        models.wren,
        backend.wren;
        latent_source=WrenCore.FromPrior(Xoshiro(seed)),
    )
    # A bit hacky to extract the internal field, but it is a vector already in the right
    # shape...
    params = wren_model.sample

    wren_value, wren_grad = WrenCore.lp_and_grad(wren_model, params)

    if backend.dppl === nothing
        reference = WrenCore.VecModel(
            models.wren,
            ADTypes.AutoForwardDiff();
            latent_source=FromPrior(Xoshiro(seed)),
        )
        reference_value, reference_grad = WrenCore.lp_and_grad(reference, params)
        @assert isapprox(wren_value, reference_value; rtol=1e-8, atol=1e-8)
        @assert isapprox(wren_grad, reference_grad; rtol=1e-7, atol=1e-8)

        WrenCore.lp(wren_model, params)
        wren_lp = median(@be WrenCore.lp($wren_model, $params) seconds=BENCHMARK_SECONDS)
        WrenCore.lp_and_grad(wren_model, params)
        wren_grad_bench =
            median(@be WrenCore.lp_and_grad($wren_model, $params) seconds=BENCHMARK_SECONDS)

        return (
            dppl_lp=nothing,
            wren_lp=wren_lp.time,
            dppl_lp_allocs=nothing,
            wren_lp_allocs=wren_lp.allocs,
            dppl_grad=nothing,
            wren_grad=wren_grad_bench.time,
            dppl_grad_allocs=nothing,
            wren_grad_allocs=wren_grad_bench.allocs,
        )
    end

    dppl_model = dppl_density(models.dppl, backend.dppl)
    dppl_value, dppl_grad = LogDensityProblems.logdensity_and_gradient(dppl_model, params)
    @assert isapprox(wren_value, dppl_value; rtol=1e-8, atol=1e-8)
    @assert isapprox(wren_grad, dppl_grad; rtol=1e-7, atol=1e-8)

    WrenCore.lp(wren_model, params)
    wren_lp = median(@be WrenCore.lp($wren_model, $params) seconds=BENCHMARK_SECONDS)
    LogDensityProblems.logdensity(dppl_model, params)
    dppl_lp = median(
        @be LogDensityProblems.logdensity($dppl_model, $params) seconds=BENCHMARK_SECONDS
    )

    WrenCore.lp_and_grad(wren_model, params)
    wren_grad_bench =
        median(@be WrenCore.lp_and_grad($wren_model, $params) seconds=BENCHMARK_SECONDS)
    LogDensityProblems.logdensity_and_gradient(dppl_model, params)
    dppl_grad_bench = median(
        @be LogDensityProblems.logdensity_and_gradient($dppl_model, $params) seconds=BENCHMARK_SECONDS
    )

    return (
        dppl_lp=dppl_lp.time,
        wren_lp=wren_lp.time,
        dppl_lp_allocs=dppl_lp.allocs,
        wren_lp_allocs=wren_lp.allocs,
        dppl_grad=dppl_grad_bench.time,
        wren_grad=wren_grad_bench.time,
        dppl_grad_allocs=dppl_grad_bench.allocs,
        wren_grad_allocs=wren_grad_bench.allocs,
    )
end

function print_header()
    println(
        rpad("model", 16),
        rpad("backend", 14),
        lpad("DPPL lp", 12),
        lpad("Wren lp", 12),
        lpad("speedup", 10),
        lpad("lp allocs", 12),
        lpad("DPPL grad", 12),
        lpad("Wren grad", 12),
        lpad("speedup", 10),
        lpad("grad allocs", 16),
    )
    println(repeat('─', 126))
end

function print_result(model_name, backend_name, result)
    if result.dppl_lp === nothing
        @printf(
            "%-16s%-14s%12s%12s%10s%12s%12s%12s%10s%16s\n",
            model_name,
            backend_name,
            "—",
            pretty_time(result.wren_lp),
            "—",
            "—/$(round(Int, result.wren_lp_allocs))",
            "—",
            pretty_time(result.wren_grad),
            "—",
            "—/$(round(Int, result.wren_grad_allocs))",
        )
        return nothing
    end
    @printf(
        "%-16s%-14s%12s%12s%9.2fx%12s%12s%12s%9.2fx%16s\n",
        model_name,
        backend_name,
        pretty_time(result.dppl_lp),
        pretty_time(result.wren_lp),
        result.dppl_lp / result.wren_lp,
        "$(round(Int, result.dppl_lp_allocs))/$(round(Int, result.wren_lp_allocs))",
        pretty_time(result.dppl_grad),
        pretty_time(result.wren_grad),
        result.dppl_grad / result.wren_grad,
        "$(round(Int, result.dppl_grad_allocs))/$(round(Int, result.wren_grad_allocs))",
    )
end

function error_summary(exception)
    message = first(Base.split(sprint(showerror, exception), '\n'))
    return length(message) > 80 ? first(message, 77) * "..." : message
end

function main()
    println("DPPL vs Wren — unconstrained vector parameters")
    println("Chairmarks sampling time per measurement: $(BENCHMARK_SECONDS) s\n")
    print_header()
    for (model_index, models) in enumerate(BenchmarkModels.models())
        for (backend_index, backend) in enumerate(BACKENDS)
            seed = 468 + 10 * model_index + backend_index
            # mnist causes Differ to stackoverflow
            # backend.wren isa Differ.AutoDifferForwards &&
            #     contains(models.name, "mnist") &&
            #     continue
            # backend.wren isa Differ.AutoDifferReverse &&
            #     contains(models.name, "mnist") &&
            #     continue
            try
                result = benchmark_pair(models, backend, seed)
                print_result(models.name, backend.name, result)
            catch exception
                @printf(
                    "%-16s%-16sERROR: %s\n",
                    models.name,
                    backend.name,
                    error_summary(exception),
                )
            end
        end
    end
    return nothing
end

main()
