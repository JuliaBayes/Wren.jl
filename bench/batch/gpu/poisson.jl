# Alternative formulations of the `BatchedPoisson` log density, for benchmarking on a GPU.
# See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/poisson.jl
#
# Every variant takes `(λ,)` as a tuple of parameter arrays and `x` as the observation,
# and returns the summed log density of independent `Poisson(λ[i])`.
# The `loggamma` term depends only on `x` (constant for the gradient), so the interesting
# part is how `xlogy(x, λ) - λ` is handled.

include("common.jl")

using LogExpFunctions: xlogy
using SpecialFunctions: loggamma

# 1. Scalar-accumulator loop using xlogy (current WrenDists implementation).
function lp_loop_xlogy((λ,), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        s += xlogy(x[i], λ[i]) - λ[i] - loggamma(x[i] + one(x[i]))
    end
    return s
end

# 2. Broadcast using xlogy.
function lp_broadcast_xlogy((λ,), x)
    return sum(@. xlogy(x, λ) - λ - loggamma(x + one(x)))
end

# 3. Branch-free: x * log(λ) directly, without xlogy's zero-guard.
#    Valid when x >= 0 and λ > 0, which holds for well-formed Poisson data.
function lp_loop_mul((λ,), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        s += x[i] * log(λ[i]) - λ[i] - loggamma(x[i] + one(x[i]))
    end
    return s
end

# 4. Broadcast, branch-free.
function lp_broadcast_mul((λ,), x)
    return sum(@. x * log(λ) - λ - loggamma(x + one(x)))
end

using WrenDists: BatchedPoisson
using DensityInterface: logdensityof

function lp_impl((λ,), x)
    @inline logdensityof(BatchedPoisson(λ), x)
end

const VARIANTS = [
    "loop_xlogy" => ScalarIndexed(lp_loop_xlogy),
    "loop_mul" => ScalarIndexed(lp_loop_mul),
    "broadcast_xlogy" => lp_broadcast_xlogy,
    "broadcast_mul" => lp_broadcast_mul,
    "impl" => lp_impl,
]

function make_inputs(rng, T, N)
    λ = exp.(T(1.0) .+ T(0.5) .* randn(rng, T, N))
    x = T.(round.(Int, λ .+ sqrt.(λ) .* randn(rng, T, N)))
    x = max.(x, zero(T))
    return (λ,), x
end

run_variants(VARIANTS, make_inputs, lp_loop_xlogy)
