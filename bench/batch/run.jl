# Benchmarks `@batch` loops against the equivalent plain `for` loops, for each of the
# WrenDists likelihoods that have specialised `WrenCore.batched` methods.
#
# Each model has a latent vector `mu` of length N and an observed vector `y`, with the
# likelihood of `y[i]` depending on `mu[i]`. The batched and looped models are evaluated
# on the same parameters, across a range of N, so that we can see where vectorisation
# starts to pay off.
#
# Run from the `bench` directory with `julia --project=. ./batch/run.jl`. `WREN_BENCH_NS`
# is a comma-separated list of N values (default 10,100,1000,10000).

using ADTypes
using Enzyme: Reverse
using Printf: @printf, @sprintf
using Random: Xoshiro
using WrenCore
using WrenDists

const NS = parse.(Int, split(get(ENV, "WREN_BENCH_NS", "10,100,1000,10000"), ","))

const ENZYME = AutoEnzyme(; mode=Reverse)

# Defines a batched and a looped model that differ only in the `@batch` annotation.
macro batch_vs_loop(name, dist_expr)
    batch_name = Symbol(name, :_batch)
    loop_name = Symbol(name, :_loop)
    return esc(quote
        @wren function $batch_name(N, y, n)
            mu ~ WrenCore.iid(WrenDists.Normal(), N)
            @batch for i in 1:N
                @observe y[i] ~ $dist_expr
            end
        end
        @wren function $loop_name(N, y, n)
            mu ~ WrenCore.iid(WrenDists.Normal(), N)
            for i in 1:N
                @observe y[i] ~ $dist_expr
            end
        end
        push!(MODELS, ($(string(name)), $batch_name, $loop_name))
    end)
end

const MODELS = Tuple{String,Any,Any}[]
@batch_vs_loop normal WrenDists.Normal(mu[i], 1.0)
@batch_vs_loop lognormal WrenDists.LogNormal(mu[i], 1.0)
@batch_vs_loop exponential WrenDists.Exponential(exp(mu[i]))
@batch_vs_loop bernoulli WrenDists.Bernoulli(1 / (1 + exp(-mu[i])))
@batch_vs_loop binomial WrenDists.Binomial(n[i], 1 / (1 + exp(-mu[i])))
@batch_vs_loop poisson WrenDists.Poisson(exp(mu[i]))

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

# Median time for one log-density-and-gradient evaluation.
function grad_time(model, adtype)
    result = WrenCore.benchmark_and_test(
        model,
        adtype;
        benchmark_model=false,
        benchmark_primal=false,
        verbose=false,
    )
    return result.grad_bench.time
end

println("Times are for one gradient evaluation\n")
@printf("%-12s %7s %14s %14s %8s\n", "model", "N", "enzyme loop", "enzyme batch", "speedup")
for (name, batch_model, loop_model) in MODELS, N in NS
    rng = Xoshiro(468)
    # Every distribution here accepts small non-negative integers as observations, so the
    # same `y` works for all of them.
    y = Float64.(rand(rng, 0:3, N))
    n = rand(rng, 3:10, N)
    t_loop = grad_time(loop_model(N, y, n), ENZYME)
    t_batch = grad_time(batch_model(N, y, n), ENZYME)
    @printf(
        "%-12s %7d %14s %14s %7.2fx\n",
        name,
        N,
        pretty_time(t_loop),
        pretty_time(t_batch),
        t_loop / t_batch,
    )
end
