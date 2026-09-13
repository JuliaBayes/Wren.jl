# Alternative formulations of the `DiagNormal` log density, for benchmarking on a GPU.
# See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/diagnormal.jl
#
# Every variant takes `(mu, sigma)` as a tuple of parameter arrays and `x` as the
# observation, and returns the summed log density of independent `Normal(mu[i], sigma[i])`.

include("common.jl")

using WrenDists: WrenDists
const log2π = WrenDists.log2π

# 1. The current WrenDists implementation: a scalar-accumulator loop, wrapped in `@trace`
#    so Reactant emits a while loop rather than unrolling it. Marked `ScalarIndexed` in
#    `VARIANTS` since it indexes traced arrays; see common.jl.
function lp_loop((mu, sigma), x)
    s = zero(eltype(x))
    @trace for i in 1:length(x)
        z = (x[i] - mu[i]) / sigma[i]
        s -= abs2(z) / 2 + log(sigma[i])
    end
    return s - length(x) * log2π / 2
end

# 2. One fused broadcast, reduced by a single `sum`.
function lp_broadcast((mu, sigma), x)
    s = sum(@. -abs2((x - mu) / sigma) / 2 - log(sigma))
    return s - length(x) * log2π / 2
end

# 3. Whiten first, then two separate reductions.
function lp_split((mu, sigma), x)
    z = (x .- mu) ./ sigma
    return -sum(abs2, z) / 2 - sum(log, sigma) - length(x) * log2π / 2
end

# 4. As 3, but multiplying by a precomputed reciprocal instead of dividing.
function lp_invsigma((mu, sigma), x)
    inv_sigma = inv.(sigma)
    z = (x .- mu) .* inv_sigma
    return -sum(abs2, z) / 2 + sum(log, inv_sigma) - length(x) * log2π / 2
end

# 5. A three-argument `mapreduce`, which may lower to a different reduction than a
#    broadcast followed by `sum`.
function lp_mapreduce((mu, sigma), x)
    s = mapreduce((xi, mi, si) -> -abs2((xi - mi) / si) / 2 - log(si), +, x, mu, sigma)
    return s - length(x) * log2π / 2
end

const VARIANTS = [
    "loop" => ScalarIndexed(lp_loop),
    "broadcast" => lp_broadcast,
    "split" => lp_split,
    "invsigma" => lp_invsigma,
    "mapreduce" => lp_mapreduce,
]

function make_inputs(rng, N)
    mu = randn(rng, N)
    sigma = exp.(0.3 .* randn(rng, N))
    x = mu .+ sigma .* randn(rng, N)
    return (mu, sigma), x
end

run_variants(VARIANTS, make_inputs, lp_loop)
