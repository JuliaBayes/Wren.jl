# Benchmarks the half distributions against equivalent Distributions.jl truncations.
# Run in a temporary environment with WrenCore and WrenDists developed from this checkout.
# See bench/adhoc/README.md for setup and https://github.com/JuliaBayes/Wren.jl/issues/128.

using ADTypes
using Chairmarks: @be
using Distributions
using Enzyme
using ForwardDiff
using Pkg
using Printf: @printf, @sprintf
using Random: Xoshiro
using Reactant
using Statistics: median
using WrenCore
using WrenDists: WrenDists as WD

const NS = parse.(Int, split(get(ENV, "WREN_BENCH_NS", "1,1000"), ","))
const SECONDS = parse(Float64, get(ENV, "WREN_BENCH_SECONDS", "1.0"))
const BACKENDS = split(get(ENV, "WREN_BENCH_BACKENDS", "enzyme,reactant"), ",")
const DEVICE = get(ENV, "WREN_BENCH_DEVICE", "auto")
const ENZYME = AutoEnzyme(; mode=Reverse)
const ADTYPES = Dict("enzyme" => ENZYME, "reactant" => AutoReactant(ENZYME))

reference(::Type{WD.HalfNormal}, mu, sigma) =
    truncated(Distributions.Normal(mu, sigma); lower=0)
reference(::Type{WD.HalfCauchy}, mu, sigma) =
    truncated(Distributions.Cauchy(mu, sigma); lower=0)
reference(::Type{WD.HalfTDist}, nu, mu, sigma) =
    truncated(mu + sigma * Distributions.TDist(nu); lower=0)

const CASES = (
    (WD.HalfNormal, (0.0, 2.0)),
    (WD.HalfNormal, (-1.5, 0.7)),
    (WD.HalfNormal, (1.0, 2.0)),
    (WD.HalfCauchy, (0.0, 2.0)),
    (WD.HalfCauchy, (-1.5, 0.7)),
    (WD.HalfCauchy, (1.0, 2.0)),
    (WD.HalfTDist, (3.0, 0.0, 2.0)),
    (WD.HalfTDist, (3.5, -1.5, 0.7)),
    (WD.HalfTDist, (40.0, 1.0, 2.0)),
)

density(d, xs) = sum(x -> logdensityof(d, x), xs)

# Keep x latent so the benchmark can differentiate its samples.
@wren function half_model(d, x)
    @iid x ~ d
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

function print_result(name, params, N, backend, operation, trial)
    b = median(trial)
    @printf(
        "%-12s %-18s %6d %-14s %-18s %12s %10d %8d\n",
        name,
        string(params),
        N,
        backend,
        operation,
        pretty_time(b.time),
        round(Int, b.bytes),
        round(Int, b.allocs),
    )
end

function benchmark_scalar(Dist, params, N, xs)
    name = string(nameof(Dist))
    d = Dist(params...)
    d_ref = reference(Dist, params...)
    @assert isapprox(density(d, xs), density(d_ref, xs); rtol=1e-8, atol=1e-8)
    wd_config = ("WrenDists", p -> Dist(p...))
    d_config = ("Distributions", p -> reference(Dist, p...))
    for (backend, construct) in (wd_config, d_config)
        # Pass the constructor's parameters as inputs so constant folding cannot hide its cost.
        dist = construct(params)
        rng = Xoshiro(468)
        operations = (
            ("construct", construct),
            ("density", _ -> density(dist, xs)),
            ("construct+density", p -> density(construct(p), xs)),
            ("rand", _ -> rand(rng, dist, N)),
        )
        for (operation, f) in operations
            f(params)
            print_result(name, params, N, backend, operation, @be $params f seconds=SECONDS)
        end
    end
end

function benchmark_model(Dist, params, N, xs)
    name = string(nameof(Dist))
    model = half_model(Dist(params...), xs)
    source = FromParams(VarNamedTuple(x=xs), nothing)
    d_ref = reference(Dist, params...)
    # Both distributions use the log transform on [0, ∞), including its Jacobian.
    reference_term(t) = logpdf(d_ref, exp(t)) + t
    z = log.(xs)
    expected = sum(reference_term, z)
    # Each observation depends on one coordinate, so scalar AD gives the full gradient.
    expected_grad = ForwardDiff.derivative.(reference_term, z)
    for backend in BACKENDS
        adtype = ADTYPES[backend]
        vm = VecModel(model, adtype; latent_source=source)
        value, grad = lp_and_grad(vm, z)
        @assert isapprox(lp(vm, z), expected; rtol=1e-8, atol=1e-8)
        @assert isapprox(value, expected; rtol=1e-8, atol=1e-8)
        @assert isapprox(grad, expected_grad; rtol=1e-7, atol=1e-8)
        for (operation, f) in (("model lp", lp), ("model lp+grad", lp_and_grad))
            trial = @be f($vm, $z) seconds=SECONDS
            print_result(name, params, N, backend, operation, trial)
        end
    end
end

function main()
    DEVICE == "auto" || Reactant.set_default_backend(DEVICE)
    println("Julia: ", VERSION, " (", Threads.nthreads(), " threads)")
    println("CPU: ", Sys.cpu_info()[1].model)
    println("Revision: ", readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
    Pkg.status()
    println("Reactant devices: ", join(Reactant.XLA.device_kind.(Reactant.devices()), ", "))
    println("Backends: ", join(BACKENDS, ", "))
    println("Seconds per measurement: ", SECONDS)
    println(
        "Times exclude setup and compilation. Reactant model calls include host transfers.\n",
    )
    @printf(
        "%-12s %-18s %6s %-14s %-18s %12s %10s %8s\n",
        "distribution",
        "parameters",
        "N",
        "backend",
        "operation",
        "time",
        "bytes",
        "allocs"
    )
    for (Dist, params) in CASES, N in NS
        xs = 0.25 .+ rand(Xoshiro(468), N)
        benchmark_scalar(Dist, params, N, xs)
        benchmark_model(Dist, params, N, xs)
    end
    return nothing
end

main()
