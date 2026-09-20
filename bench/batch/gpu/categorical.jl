# Alternative formulations of a batched Categorical log density, for benchmarking on a GPU.
# See common.jl for the harness. Run from the `bench` directory with
#
#     julia --project=. ./batch/gpu/categorical.jl
#
# Unlike other batched distributions whose parameters are vectors (one scalar per sample),
# Categorical's parameter is a probability vector per sample, so the batched parameter is a
# matrix P of shape (K, N) — categories along rows, batch elements along columns.
#
# Every variant takes `(P,)` as a tuple of parameter arrays and `x` as an integer
# observation vector (values in 1:K), and returns the summed log density.
# The gradient is with respect to `P`.
#
# `WREN_BENCH_KS` is a comma-separated list of K (number of categories) values
# (default 4,32).

include("common.jl")

const KS = parse.(Int, split(get(ENV, "WREN_BENCH_KS", "4,32"), ","))

# --- Variants ---

# 1. Scalar loop: gather P[x[i], i] one element at a time.
function lp_loop((P,), x)
    s = zero(eltype(P))
    @trace for i in 1:length(x)
        s += log(P[x[i], i])
    end
    return s
end

# 2. Broadcast with CartesianIndex gather.
function lp_broadcast_cartesian((P,), x)
    N = length(x)
    idx = CartesianIndex.(x, 1:N)
    return sum(log, @view P[idx])
end

# 3. One-hot via broadcast comparison (no scalar indexing for construction).
function lp_onehot_broadcast((P,), x)
    K = size(P, 1)
    T = eltype(P)
    categories = T.(1:K)  # (K,)
    H = categories .== reshape(x, 1, :)  # (K, N) Bool matrix
    return sum(log.(P) .* H)
end

# 4. Loop over categories with @trace inner loop for Reactant compatibility.
function lp_category_loop_trace((P,), x)
    K, N = size(P)
    T = eltype(P)
    s = zero(T)
    for k in 1:K
        @trace for i in 1:N
            s += ifelse(x[i] == T(k), log(P[k, i]), zero(T))
        end
    end
    return s
end

# 5. Broadcast: sum over k of (x .== k) .* log(P[k, :]) — fully vectorised.
function lp_category_broadcast((P,), x)
    K = size(P, 1)
    T = eltype(P)
    s = zero(T)
    for k in 1:K
        s += sum((x .== T(k)) .* log.(@view(P[k, :])))
    end
    return s
end

# --- Harness ---

const VARIANTS = [
    "loop" => ScalarIndexed(lp_loop),
    "bcast_cartesian" => lp_broadcast_cartesian,
    "onehot_bcast" => lp_onehot_broadcast,
    "cat_loop_trace" => ScalarIndexed(lp_category_loop_trace),
    "cat_broadcast" => lp_category_broadcast,
]

function make_inputs(rng, T, N, K)
    # Random probability matrix, each column sums to 1
    raw = rand(rng, T, K, N) .+ T(0.01)
    P = raw ./ sum(raw; dims=1)
    x = rand(rng, 1:K, N)
    return (P,), x
end

# Override the default run to sweep over K as well as N.
function main()
    print_status()
    println("K values:         ", join(KS, ", "))
    println()
    println("Times are for one evaluation of the primal, and one of value-and-gradient\n")
    @printf("%-20s %4s %9s", "variant", "K", "N")
    for backend in BACKENDS
        primal_str = backend == "enzyme" ? "primal" : "$backend primal"
        @printf(" %16s %16s", primal_str, backend * " grad")
    end
    println()

    for K in KS, N in NS
        params, x = make_inputs(Xoshiro(468), ELTYPE, N, K)
        ref_f = _unwrap(VARIANTS[1].second)
        ref_val, ref_grad = ValueAndGrad(ref_f)(params, x)
        for (name, variant) in VARIANTS
            @printf("%-20s %4d %9d", name, K, N)
            for backend in BACKENDS
                times = try
                    TIMERS[backend](name, N, variant, params, x, ref_val, ref_grad)
                catch err
                    msg = first(split(sprint(showerror, err), '\n'))
                    @warn "$backend failed on $name at K=$K, N=$N: $msg"
                    nothing
                end
                if isnothing(times)
                    @printf(" %16s %16s", "error", "error")
                else
                    @printf(" %16s %16s", pretty_time(times[1]), pretty_time(times[2]))
                end
            end
            println()
        end
        println()
    end
end

main()
