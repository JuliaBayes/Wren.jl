# Benchmarks a dedicated shared-sigma batched normal (`IsoNormal`, a scalar `sigma`) against
# a batched normal that stores the shared `sigma` as a `Fill` array.
# Run in a temporary environment with WrenCore and WrenDists developed from this checkout.
# See bench/adhoc/README.md for setup.

using ADTypes
using Chairmarks: @be
using Distributions: Distributions, logpdf
using Enzyme
using FillArrays: Fill
using Pkg
using Printf: @printf, @sprintf
using Random: Random, Xoshiro
using Reactant
using Statistics: median
using WrenCore
using WrenDists: WrenDists as WD

const NS = parse.(Int, split(get(ENV, "WREN_BENCH_NS", "10,1000,100000"), ","))
const SECONDS = parse(Float64, get(ENV, "WREN_BENCH_SECONDS", "1.0"))
const BACKENDS = split(get(ENV, "WREN_BENCH_BACKENDS", "enzyme,reactant"), ",")
const DEVICE = get(ENV, "WREN_BENCH_DEVICE", "auto")
const ENZYME = AutoEnzyme(; mode=Reverse)
const ADTYPES = Dict("enzyme" => ENZYME, "reactant" => AutoReactant(ENZYME))
const SIGMA = 2.0

# `IsoNormal` and `DiagNormal` are not part of WrenDists. They are defined here so that the
# scalar-sigma type and the elementwise log density can be timed next to `BatchedNormal`.

struct IsoNormal{M<:AbstractArray,S<:Number} <: WD.BatchedUnivariate
    mu::M
    sigma::S
end
Base.size(d::IsoNormal) = size(d.mu)
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .* d.sigma .+ d.mu
function _isonormal_sqdist(d::IsoNormal, x)
    dist = WD._zero_logdensity(d, x)
    @inbounds for i in eachindex(x, d.mu)
        dist += abs2(x[i] - d.mu[i])
    end
    return dist
end
function WD.logdensityof(d::IsoNormal, x)
    # Reactant does not allow scalar indexing, so it gets the broadcasted version.
    dist = if WD.within_compile()
        sum(abs2, x .- d.mu)
    else
        _isonormal_sqdist(d, x)
    end
    N = length(x)
    return -0.5 * dist / abs2(d.sigma) - N * log(d.sigma) - 0.5 * N * WD.log2π
end

# Takes one `log` and one division per element, whatever the array type of `sigma`.
struct DiagNormal{M<:AbstractArray,S<:AbstractArray} <: WD.BatchedUnivariate
    mu::M
    sigma::S
end
Base.size(d::DiagNormal) = size(d.mu)
Base.rand(rng::Random.AbstractRNG, d::DiagNormal) =
    randn(rng, size(d.mu)) .* d.sigma .+ d.mu
function WD.logdensityof(d::DiagNormal, x::AbstractArray{<:Number})
    mu, sigma = d.mu, d.sigma
    T = float(promote_type(eltype(x), eltype(mu), eltype(sigma)))
    lognorm = length(x) * convert(T, WD.log2π) / 2
    return if WD.within_compile()
        z = (x .- mu) ./ sigma
        -sum(abs2, z) / 2 - sum(log, sigma) - lognorm
    else
        s = zero(T)
        for i in eachindex(x)
            z = (x[i] - mu[i]) / sigma[i]
            s -= abs2(z) / 2 + log(sigma[i])
        end
        s - lognorm
    end
end

# A model argument that is a function has a concrete type; the `IsoNormal` type itself is a
# `UnionAll`, which makes the model call it through dynamic dispatch.
iso(mu, sigma) = IsoNormal(mu, sigma)
diag_fill(mu, sigma) = DiagNormal(mu, Fill(sigma, size(mu)))
batched_fill(mu, sigma) = WD.BatchedNormal(mu, Fill(sigma, size(mu)))

# The first candidate is the baseline of the `vs IsoNormal` column. Timing it again at the
# end shows how far two measurements of the same code drift apart.
const CANDIDATES = (
    ("IsoNormal", iso),
    ("DiagNormal+Fill", diag_fill),
    ("BatchedNormal+Fill", batched_fill),
    ("IsoNormal (repeat)", iso),
)

const DIRECT_OPERATIONS = (
    ("construct", (construct, dist, params, x, rng) -> construct(params...)),
    ("density", (construct, dist, params, x, rng) -> logdensityof(dist, x)),
    (
        "construct+density",
        (construct, dist, params, x, rng) -> logdensityof(construct(params...), x),
    ),
    ("rand", (construct, dist, params, x, rng) -> rand(rng, dist)),
)

@wren function sigma_constant_model(construct, x, sigma)
    mu ~ WrenCore.iid(WD.Normal(), length(x))
    @observe x ~ construct(mu, sigma)
end

@wren function sigma_latent_model(construct, x)
    mu ~ WrenCore.iid(WD.Normal(), length(x))
    sigma ~ WD.Exponential()
    @observe x ~ construct(mu, sigma)
end

likelihood(mu, sigma, x) = sum(logpdf.(Distributions.Normal.(mu, sigma), x))

# Each reference returns the log density and its gradient at the unconstrained vector `z`.
function reference_sigma_constant(z, x)
    mu = z
    value = sum(logpdf.(Distributions.Normal(), mu)) + likelihood(mu, SIGMA, x)
    grad = @. -mu + (x - mu) / SIGMA^2
    return value, grad
end

# `sigma` uses the log transform on [0, ∞), including its Jacobian.
function reference_sigma_latent(z, x)
    mu, log_sigma = z[begin:(end-1)], z[end]
    sigma = exp(log_sigma)
    value =
        sum(logpdf.(Distributions.Normal(), mu)) +
        logpdf(Distributions.Exponential(), sigma) +
        log_sigma +
        likelihood(mu, sigma, x)
    grad_mu = @. -mu + (x - mu) / sigma^2
    grad_log_sigma = -sigma + 1 - length(x) + sum(abs2, x .- mu) / sigma^2
    return value, vcat(grad_mu, grad_log_sigma)
end

const MODEL_CASES = (
    (
        "sigma constant",
        (construct, x) -> sigma_constant_model(construct, x, SIGMA),
        mu -> mu,
        reference_sigma_constant,
    ),
    (
        "sigma latent",
        sigma_latent_model,
        mu -> vcat(mu, log(SIGMA)),
        reference_sigma_latent,
    ),
)

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

function print_result(case, N, backend, operation, candidate, trial, baseline_time)
    b = median(trial)
    @printf(
        "%-15s %7d %-9s %-18s %-19s %12s %12.2f %10d %8d\n",
        case,
        N,
        backend,
        operation,
        candidate,
        pretty_time(b.time),
        b.time / baseline_time,
        round(Int, b.bytes),
        round(Int, b.allocs),
    )
end

# `measure(construct)` returns the trial of one candidate. Candidates of one operation are
# timed back to back so that they share machine conditions.
function benchmark_candidates(measure, case, N, backend, operation)
    baseline_time = nothing
    for (candidate, construct) in CANDIDATES
        trial = measure(construct)
        baseline_time = something(baseline_time, median(trial).time)
        print_result(case, N, backend, operation, candidate, trial, baseline_time)
    end
end

function benchmark_direct(N, mu, x)
    # Pass the constructor's parameters as inputs so constant folding cannot hide its cost.
    params = (mu, SIGMA)
    expected = likelihood(mu, SIGMA, x)
    for (_, construct) in CANDIDATES
        @assert isapprox(logdensityof(construct(params...), x), expected; rtol=1e-8)
    end
    for (operation, f) in DIRECT_OPERATIONS
        benchmark_candidates("direct", N, "none", operation) do construct
            dist = construct(params...)
            rng = Xoshiro(468)
            f(construct, dist, params, x, rng)
            @be $f($construct, $dist, $params, $x, $rng) seconds=SECONDS
        end
    end
end

function checked_vecmodel(model, adtype, z, expected, expected_grad)
    vm = VecModel(model, adtype)
    value, grad = lp_and_grad(vm, z)
    @assert isapprox(lp(vm, z), expected; rtol=1e-8)
    @assert isapprox(value, expected; rtol=1e-8)
    @assert isapprox(grad, expected_grad; rtol=1e-7, atol=1e-8)
    return vm
end

function benchmark_model(N, mu, x)
    for (case, make_model, unconstrained, reference) in MODEL_CASES
        z = unconstrained(mu)
        expected, expected_grad = reference(z, x)
        for backend in BACKENDS
            adtype = ADTYPES[backend]
            vecmodels = Dict(
                construct => checked_vecmodel(
                    make_model(construct, x),
                    adtype,
                    z,
                    expected,
                    expected_grad,
                ) for construct in unique(last.(CANDIDATES))
            )
            for (operation, f) in (("model lp", lp), ("model lp+grad", lp_and_grad))
                benchmark_candidates(case, N, backend, operation) do construct
                    vm = vecmodels[construct]
                    @be $f($vm, $z) seconds=SECONDS
                end
            end
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
        "%-15s %7s %-9s %-18s %-19s %12s %12s %10s %8s\n",
        "case",
        "N",
        "backend",
        "operation",
        "candidate",
        "time",
        "vs IsoNormal",
        "bytes",
        "allocs"
    )
    for N in NS
        rng = Xoshiro(468)
        mu = randn(rng, N)
        x = mu .+ SIGMA .* randn(rng, N)
        benchmark_direct(N, mu, x)
        benchmark_model(N, mu, x)
    end
    return nothing
end

main()
