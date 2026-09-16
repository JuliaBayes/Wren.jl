# Alternative formulations of the `BatchedBinomial` log density, for benchmarking on a GPU.
# See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/binomial.jl
#
# Every variant takes `(n, p)` as a tuple of parameter arrays and `x` as the observation,
# and returns the summed log density of independent `Binomial(n[i], p[i])`.
# The gradient is with respect to both `n` and `p`. The `logbinom` term depends on `n` and
# `x`; the `xlogy` terms depend on `x`, `n`, and `p`.

include("common.jl")

using LogExpFunctions: xlogy
using SpecialFunctions: loggamma

function logbinom(n::T, k::T) where {T}
    nc, kc, mc = max(n, zero(T)), max(k, zero(T)), max(n - k, zero(T))
    return loggamma(nc + one(T)) - loggamma(kc + one(T)) - loggamma(mc + one(T))
end

# 1. Scalar-accumulator loop using xlogy (current WrenDists implementation).
function lp_loop_xlogy((n, p), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        T = eltype(x)
        ni, ki, prob = convert(T, n[i]), convert(T, x[i]), convert(T, p[i])
        s += logbinom(ni, ki) + xlogy(ki, prob) + xlogy(ni - ki, one(T) - prob)
    end
    return s
end

# 2. Broadcast using xlogy.
function lp_broadcast_xlogy((n, p), x)
    return sum(@. logbinom(n, x) + xlogy(x, p) + xlogy(n - x, one(x) - p))
end

# 3. Branch-free: k * log(p) + (n-k) * log1p(-p) directly instead of xlogy.
#    Valid when 0 < p < 1 and 0 <= k <= n, which holds for well-formed Binomial data
#    with non-degenerate p.
function lp_loop_mul((n, p), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        T = eltype(x)
        ni, ki, prob = convert(T, n[i]), convert(T, x[i]), convert(T, p[i])
        s += logbinom(ni, ki) + ki * log(prob) + (ni - ki) * log1p(-prob)
    end
    return s
end

# 4. Broadcast, branch-free.
function lp_broadcast_mul((n, p), x)
    return sum(@. logbinom(n, x) + x * log(p) + (n - x) * log1p(-p))
end

using WrenDists: BatchedBinomial
using DensityInterface: logdensityof

function lp_impl((n, p), x)
    @inline logdensityof(BatchedBinomial(n, p), x)
end

const VARIANTS = [
    "loop_xlogy" => ScalarIndexed(lp_loop_xlogy),
    "loop_mul" => ScalarIndexed(lp_loop_mul),
    "broadcast_xlogy" => lp_broadcast_xlogy,
    "broadcast_mul" => lp_broadcast_mul,
    "impl" => lp_impl,
]

function make_inputs(rng, T, N)
    n = T.(rand(rng, 5:50, N))
    p = T(0.1) .+ T(0.8) .* rand(rng, T, N)
    x = T.(clamp.(round.(Int, n .* p .+ sqrt.(n .* p .* (1 .- p)) .* randn(rng, T, N)), 0, 100))
    x = min.(x, n)
    return (n, p), x
end

run_variants(VARIANTS, make_inputs, lp_loop_xlogy)
