# Alternative formulations of the `BatchedBernoulli` log density, for benchmarking on a
# GPU. See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/bernoulli.jl
#
# Every variant takes `(p,)` as a tuple of parameter arrays and `x` as the observation,
# and returns the summed log density of independent `Bernoulli(p[i])`.
# The key question is whether the data-dependent branch in the ifelse formulation
# prevents XLA from flattening the @trace for loop the way it does for BatchedNormal.

include("common.jl")

# 1. The current WrenDists implementation: ifelse chain inside a @trace for loop.
function lp_loop((p,), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        xi = x[i]
        prob = p[i]
        s += ifelse(
            xi == one(xi),
            log(prob),
            ifelse(xi == zero(xi), log1p(-prob), oftype(prob, -Inf)),
        )
    end
    return s
end

# 2. Same ifelse arithmetic, but as a broadcast + sum.
function lp_broadcast_ifelse((p,), x)
    return sum(
        @. ifelse(x == one(x), log(p), ifelse(x == zero(x), log1p(-p), oftype(p, -Inf)))
    )
end

# 3. Branch-free: x * log(p) + (1 - x) * log1p(-p). Always computes both terms,
#    but avoids selects. Only valid for x in {0, 1}.
function lp_broadcast_mul((p,), x)
    return sum(@. x * log(p) + (one(x) - x) * log1p(-p))
end

# 4. Branch-free inside a @trace for loop.
function lp_loop_mul((p,), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        xi = x[i]
        prob = p[i]
        s += xi * log(prob) + (one(xi) - xi) * log1p(-prob)
    end
    return s
end

using WrenDists: BatchedBernoulli
using DensityInterface: logdensityof

function lp_impl((p,), x)
    @inline logdensityof(BatchedBernoulli(p), x)
end

const VARIANTS = [
    "loop_ifelse" => ScalarIndexed(lp_loop),
    "loop_mul" => ScalarIndexed(lp_loop_mul),
    "broadcast_ifelse" => lp_broadcast_ifelse,
    "broadcast_mul" => lp_broadcast_mul,
    "impl" => lp_impl,
]

function make_inputs(rng, T, N)
    # p values in (0, 1), avoiding exact 0/1 to keep log/log1p finite
    p = T(0.1) .+ T(0.8) .* rand(rng, T, N)
    x = T.(rand(rng, N) .< p)
    return (p,), x
end

run_variants(VARIANTS, make_inputs, lp_loop)
