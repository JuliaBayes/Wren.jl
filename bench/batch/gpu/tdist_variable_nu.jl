# Alternative formulations of the batched TDist log density with per-element ν,
# for benchmarking on a GPU. See common.jl for the harness. Run from the `bench`
# directory with
#
#     julia --project=. ./batch/gpu/tdist_variable_nu.jl
#
# Every variant takes `(ν, μ, σ)` as a tuple of parameter arrays and `x` as the
# observation, and returns the summed log density of independent `TDist(ν[i], μ[i], σ[i])`.
# Unlike tdist.jl, ν varies per element, so no terms can be hoisted.
#
# See https://github.com/JuliaBayes/Wren.jl/issues/117

include("common.jl")

using WrenDists: loggamma, logπ

function _tlognorm_sum(ν::AbstractArray)
    T = float(eltype(ν))
    return sum(ν) do νi
        νi = convert(T, νi)
        half = (νi + one(T)) / 2
        loggamma(half) - loggamma(νi / 2) - (log(νi) + T(logπ)) / 2
    end
end

# 1. Scalar-accumulator loop.
function lp_loop(params, x)
    ν, μ, σ = params
    T = float(promote_type(eltype(x), eltype(ν), eltype(μ), eltype(σ)))
    one_T = one(T)
    s = zero(T)
    @trace for i in 1:length(x)
        νi = convert(T, ν[i])
        half = (νi + one_T) / 2
        z = (x[i] - μ[i]) / σ[i]
        s -= log(σ[i]) + half * log(one_T + z^2 / νi)
    end
    return s + _tlognorm_sum(ν)
end

# 2. Broadcast.
function lp_broadcast(params, x)
    ν, μ, σ = params
    T = float(promote_type(eltype(x), eltype(ν), eltype(μ), eltype(σ)))
    one_T = one(T)
    half = @. (T(ν) + one_T) / 2
    z = @. (x - μ) / σ
    return sum(@. -log(σ) - half * log(one_T + z^2 / ν)) + _tlognorm_sum(ν)
end

# 3. Split reductions.
function lp_split(params, x)
    ν, μ, σ = params
    T = float(promote_type(eltype(x), eltype(ν), eltype(μ), eltype(σ)))
    one_T = one(T)
    z = @. (x - μ) / σ
    half = @. (T(ν) + one_T) / 2
    return -sum(log, σ) - sum(@. half * log(one_T + z^2 / ν)) + _tlognorm_sum(ν)
end

const VARIANTS =
    ["loop" => ScalarIndexed(lp_loop), "broadcast" => lp_broadcast, "split" => lp_split]

function make_inputs(rng, T, N)
    ν = T(3) .+ rand(rng, T, N)
    μ = randn(rng, T, N)
    σ = exp.(T(0.3) .* randn(rng, T, N))
    x = μ .+ σ .* randn(rng, T, N)
    return (ν, μ, σ), x
end

run_variants(VARIANTS, make_inputs, lp_loop)
