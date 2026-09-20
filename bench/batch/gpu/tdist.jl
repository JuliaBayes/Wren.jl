# Alternative formulations of the batched TDist log density, for benchmarking on a
# GPU. See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/tdist.jl
#
# Every variant takes `(ν, μ, σ)` as a tuple of parameter arrays and `x` as the
# observation, and returns the summed log density of independent `TDist(ν[i], μ[i], σ[i])`.
# ν is a Fill array (shared across all elements), so the loggamma normalising terms
# are computed once.

include("common.jl")

using WrenDists: loggamma, logπ
using FillArrays: Fill, getindex_value

# --- Helper: normalising constant ---
# For shared ν (Fill), compute once; for varying ν, compute per element.

function _tlognorm_sum(ν::Fill)
    T = float(eltype(ν))
    ν1 = convert(T, getindex_value(ν))
    half = (ν1 + one(T)) / 2
    return length(ν) * (loggamma(half) - loggamma(ν1 / 2) - (log(ν1) + T(logπ)) / 2)
end

function _tlognorm_sum(ν::AbstractArray)
    T = float(eltype(ν))
    return sum(ν) do νi
        νi = convert(T, νi)
        half = (νi + one(T)) / 2
        loggamma(half) - loggamma(νi / 2) - (log(νi) + T(logπ)) / 2
    end
end

# 1. Scalar-accumulator loop.
function lp_loop((ν, μ, σ)::Tuple{Fill,Any,Any}, x)
    T = float(promote_type(eltype(x), eltype(ν), eltype(μ), eltype(σ)))
    N = length(x)
    ν1 = convert(T, getindex_value(ν))
    half = (ν1 + one(T)) / 2
    s = zero(T)
    @trace for i in 1:N
        z = (x[i] - μ[i]) / σ[i]
        s -= log(σ[i]) + half * log(one(T) + z^2 / ν1)
    end
    return s + _tlognorm_sum(ν)
end

# 2. Broadcast.
function lp_broadcast((ν, μ, σ)::Tuple{Fill,Any,Any}, x)
    T = float(promote_type(eltype(x), eltype(ν), eltype(μ), eltype(σ)))
    N = length(x)
    ν1 = convert(T, getindex_value(ν))
    half = (ν1 + one(T)) / 2
    z = @. (x - μ) / σ
    return sum(@. -log(σ) - half * log(one(T) + z^2 / ν1)) + _tlognorm_sum(ν)
end

# 3. Split reductions.
function lp_split((ν, μ, σ)::Tuple{Fill,Any,Any}, x)
    T = float(promote_type(eltype(x), eltype(ν), eltype(μ), eltype(σ)))
    N = length(x)
    ν1 = convert(T, getindex_value(ν))
    half = (ν1 + one(T)) / 2
    z = @. (x - μ) / σ
    one_T = one(T)
    return -sum(log, σ) - half * sum(t -> log(one_T + t^2 / ν1), z) + _tlognorm_sum(ν)
end

const VARIANTS =
    ["loop" => ScalarIndexed(lp_loop), "broadcast" => lp_broadcast, "split" => lp_split]

function make_inputs(rng, T, N)
    ν = Fill(T(4.0), N)
    μ = randn(rng, T, N)
    σ = exp.(T(0.3) .* randn(rng, T, N))
    x = μ .+ σ .* randn(rng, T, N)
    return (ν, μ, σ), x
end

run_variants(VARIANTS, make_inputs, lp_loop)
