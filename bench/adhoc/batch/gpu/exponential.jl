# Alternative formulations of the `BatchedExponential` log density, for benchmarking on a
# GPU. See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/exponential.jl
#
# Every variant takes `(θ,)` as a tuple of parameter arrays and `x` as the observation,
# and returns the summed log density of independent `Exponential(θ[i])`.

include("common.jl")

# 1. Scalar-accumulator loop (current WrenDists implementation).
function lp_loop((θ,), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        s -= x[i] / θ[i] + log(θ[i])
    end
    return s
end

# 2. Single fused broadcast.
function lp_broadcast((θ,), x)
    return sum(@. -(x / θ) - log(θ))
end

# 3. Precompute reciprocal, two separate reductions.
function lp_split((θ,), x)
    inv_θ = inv.(θ)
    return -sum(x .* inv_θ) - sum(log, θ)
end

using WrenDists: BatchedExponential
using DensityInterface: logdensityof

function lp_impl((θ,), x)
    @inline logdensityof(BatchedExponential(θ), x)
end

const VARIANTS = [
    "loop" => ScalarIndexed(lp_loop),
    "broadcast" => lp_broadcast,
    "split" => lp_split,
    "impl" => lp_impl,
]

function make_inputs(rng, T, N)
    θ = exp.(T(0.3) .* randn(rng, T, N))
    x = θ .* -log.(rand(rng, T, N))
    return (θ,), x
end

run_variants(VARIANTS, make_inputs, lp_loop)
