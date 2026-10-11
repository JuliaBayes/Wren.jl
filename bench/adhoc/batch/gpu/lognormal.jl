# Alternative formulations of the `BatchedLogNormal` log density, for benchmarking on a
# GPU. See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/lognormal.jl
#
# Every variant takes `(μ, σ)` as a tuple of parameter arrays and `x` as the observation,
# and returns the summed log density of independent `LogNormal(μ[i], σ[i])`.

include("common.jl")

using WrenDists: WrenDists

_lognorm(x) = length(x) * eltype(x)(WrenDists.log2π) / 2

# 1. Scalar-accumulator loop (current WrenDists implementation).
function lp_loop((mu, sigma), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        lx = log(x[i])
        z = (lx - mu[i]) / sigma[i]
        s -= abs2(z) / 2 + log(sigma[i]) + lx
    end
    return s - _lognorm(x)
end

# 2. Single fused broadcast.
function lp_broadcast((mu, sigma), x)
    s = sum(@. -(abs2((log(x) - mu) / sigma)) / 2 - log(sigma) - log(x))
    return s - _lognorm(x)
end

# 3. Compute log(x) once, then two reductions.
function lp_split((mu, sigma), x)
    lx = log.(x)
    z = (lx .- mu) ./ sigma
    return -sum(abs2, z) / 2 - sum(log, sigma) - sum(lx) - _lognorm(x)
end

using WrenDists: BatchedLogNormal
using DensityInterface: logdensityof

function lp_impl((mu, sigma), x)
    @inline logdensityof(BatchedLogNormal(mu, sigma), x)
end

const VARIANTS = [
    "loop" => ScalarIndexed(lp_loop),
    "broadcast" => lp_broadcast,
    "split" => lp_split,
    "impl" => lp_impl,
]

function make_inputs(rng, T, N)
    mu = randn(rng, T, N)
    sigma = exp.(T(0.3) .* randn(rng, T, N))
    x = exp.(mu .+ sigma .* randn(rng, T, N))
    return (mu, sigma), x
end

run_variants(VARIANTS, make_inputs, lp_loop)
