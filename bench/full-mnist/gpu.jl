# Compares formulations of the MNIST multinomial logistic regression model on the full
# MNIST training set (60,000 images, raw pixels):
#
#   1. DPPL, as written in `bench/adhoc/basic/models.jl` (y passed as an argument)
#   2. DPPL, with y created inside the model and supplied by conditioning
#   3. DPPL, with the observations replaced by a vectorised likelihood in `@addlogprob!`
#   4. Wren, plain `for` loop over observations
#   5. Wren, `@batch` loop over observations
#   6. Wren, with the observations replaced by the same vectorised likelihood in
#      `@addloglikelihood`
#
# All six are timed with Enzyme reverse mode on the CPU. Variants 5 and 6 are also timed
# with Reactant, on whichever device it picks (a GPU, if there is one). Variant 4 is not
# run with Reactant because Reactant would have to trace all 60,000 loop iterations one by
# one. DPPL can't be run with Reactant.
#
# Run from this folder with:
#
#     julia --project=. gpu.jl
#
# Environment variables:
#
#   - WREN_BENCH_SECONDS: Chairmarks sampling time per measurement (default 5.0)
#   - WREN_BENCH_CPU: set to `false` to skip the CPU Enzyme timings, apart from the
#     single evaluation that the Reactant results are checked against (default `true`)
#
# Float32 vs Float64
# ------------------
#
# The float type is taken from the log-probability type preference of WrenCore and
# DynamicPPL, which must agree. The data, the priors, and the parameters all use that type.
# Because these preferences are compile-time constants, you need one Julia session to set
# them and a fresh one to run the benchmark. To switch to Float32, run from this folder:
#
#     julia --project=. -e 'using WrenCore, DynamicPPL; WrenCore.set_logprob_type!(Float32); DynamicPPL.set_logprob_type!(Float32)'
#
# and then run the benchmark as above. To switch back, do the same with Float64. (This
# writes the preferences to `LocalPreferences.toml` in this folder.)
#
# Consumer GPUs are often much slower at Float64 than Float32, so the Float64 GPU timings
# will look a lot better on data-centre GPUs.

using ADTypes: AutoEnzyme, AutoReactant
using Chairmarks: @be
using DifferentiationInterface # Needed for DynamicPPL to use AD backends.
using Distributions
using DynamicPPL: DynamicPPL, filldist, @addlogprob!
using Enzyme
using LogDensityProblems
using LogExpFunctions: softmax, logsumexp
using MLDatasets: MNIST
using Printf
using Random: Xoshiro
using Reactant
using Statistics: median
using WrenCore
using WrenDists: WrenDists as WD

# MLDatasets otherwise asks for confirmation before downloading.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

const SECONDS = parse(Float64, get(ENV, "WREN_BENCH_SECONDS", "5.0"))
const RUN_CPU = parse(Bool, get(ENV, "WREN_BENCH_CPU", "true"))

const T = WrenCore.LogProbType
if DynamicPPL.LogProbType !== T
    error(
        "WrenCore's log-probability type is $T but DynamicPPL's is " *
        "$(DynamicPPL.LogProbType). See the comments at the top of this file for how to " *
        "set them.",
    )
end
T in (Float32, Float64) || error("This script only supports Float32 and Float64, not $T")

const mnist_data = MNIST(split=:train)
# N x 784 matrix of pixel values in [0, 1], one row per image
const X = Matrix(T.(reshape(mnist_data.features, 28 * 28, :))')
# Convert labels from 0-9 to 1-10 for 1-based indexing in Julia
const LABELS = mnist_data.targets .+ 1
const C = 10 # Number of labels

function logits_of(W, b, X)
    N = size(X, 1)
    nonreference_logits = W * X' .+ b # (C-1) x N
    return vcat(zeros(eltype(nonreference_logits), 1, N), nonreference_logits)
end

# Vectorised form of sum_i log(softmax(logits[:, i])[y[i]])
# This uses linear indices because Reactant can't trace `CartesianIndex.(y, 1:N)`.
function categorical_loglik(logits, y)
    K, N = size(logits)
    return sum(logits[y .+ K .* (0:(N-1))]) - sum(logsumexp(logits; dims=1))
end

# 1. DPPL as written
DynamicPPL.@model function dppl_arg(X, y, C)
    N, D = size(X)
    W ~ filldist(Normal(zero(eltype(X)), one(eltype(X))), C - 1, D)
    b ~ filldist(Normal(zero(eltype(X)), one(eltype(X))), C - 1)
    probabilities = softmax(logits_of(W, b, X); dims=1)
    for i in 1:N
        y[i] ~ Categorical(probabilities[:, i])
    end
end

# 2. DPPL, y created inside the model and conditioned on
DynamicPPL.@model function dppl_cond(X, C)
    N, D = size(X)
    W ~ filldist(Normal(zero(eltype(X)), one(eltype(X))), C - 1, D)
    b ~ filldist(Normal(zero(eltype(X)), one(eltype(X))), C - 1)
    probabilities = softmax(logits_of(W, b, X); dims=1)
    y = Vector{Int}(undef, N)
    for i in 1:N
        y[i] ~ Categorical(probabilities[:, i])
    end
end

# 3. DPPL, vectorised likelihood via @addlogprob!
DynamicPPL.@model function dppl_addlogprob(X, y, C)
    N, D = size(X)
    W ~ filldist(Normal(zero(eltype(X)), one(eltype(X))), C - 1, D)
    b ~ filldist(Normal(zero(eltype(X)), one(eltype(X))), C - 1)
    @addlogprob! (; loglikelihood=categorical_loglik(logits_of(W, b, X), y))
end

# 4. Wren, no @batch
@wren function wren_nobatch(X, y, C)
    N, D = size(X)
    @iid W ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1, D)
    @iid b ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1)
    probabilities = softmax(logits_of(W, b, X); dims=1)
    for i in 1:N
        @observe y[i] ~ WD.Categorical(probabilities[:, i])
    end
end

# 5. Wren, @batch
@wren function wren_batch(X, y, C)
    N, D = size(X)
    @iid W ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1, D)
    @iid b ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1)
    probabilities = softmax(logits_of(W, b, X); dims=1)
    @batch for i in 1:N
        @observe y[i] ~ WD.Categorical(probabilities[:, i])
    end
end

# 6. Wren, vectorised likelihood via @addloglikelihood
@wren function wren_addll(X, y, C)
    N, D = size(X)
    @iid W ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1, D)
    @iid b ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1)
    @addloglikelihood categorical_loglik(logits_of(W, b, X), y)
end

const DPPL_AD = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
const WREN_AD = AutoEnzyme(; mode=Reverse)
const REACTANT_AD = AutoReactant(AutoEnzyme(; mode=Reverse))

function dppl_ldf(model)
    return DynamicPPL.LogDensityFunction(
        model,
        DynamicPPL.getlogjoint_internal,
        DynamicPPL.LinkAll();
        adtype=DPPL_AD,
    )
end

function bench(f, obj, params)
    # Call once first, so that it's compiled before timing.
    f(obj, params)
    return median(@be f($obj, $params) seconds=SECONDS)
end

function print_header()
    @printf("%-34s%14s%10s%14s%10s\n", "variant", "lp", "allocs", "lp+grad", "allocs")
    println(repeat('─', 82))
end

function print_row(name, r_lp, r_grad)
    @printf(
        "%-34s%11.2f ms%10d%11.2f ms%10d\n",
        name,
        r_lp.time * 1e3,
        round(Int, r_lp.allocs),
        r_grad.time * 1e3,
        round(Int, r_grad.allocs),
    )
end

print_error(name, e) =
    @printf("%-34sERROR: %s\n", name, first(split(sprint(showerror, e), '\n')))

maxreldiff(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-12))

# Different variants add things up in different orders, so in Float32 they only agree to
# about 1e-5.
const LP_RTOL = T === Float64 ? 1e-8 : 1e-4
const GRAD_RTOL = T === Float64 ? 1e-6 : 1e-3

# Warn (rather than error) on a mismatch, so that one bad result doesn't throw away a
# long run.
function check(name, v, g, ref_lp, ref_grad)
    if !isapprox(v, ref_lp; rtol=LP_RTOL) || !isapprox(g, ref_grad; rtol=GRAD_RTOL)
        @warn "$name does not match the reference" v ref_lp maxreldiff(g, ref_grad)
    end
end

function main()
    println("Reactant device: ", string(first(Reactant.devices())))
    println("Float type: ", T)
    println("Julia threads: ", Threads.nthreads())
    println("Chairmarks sampling time per measurement: $(SECONDS) s\n")

    # Reference values that every other variant is checked against.
    wren_ref =
        VecModel(wren_batch(X, LABELS, C), WREN_AD; latent_source=FromPrior(Xoshiro(468)))
    params = wren_ref.sample
    ref_lp, ref_grad = lp_and_grad(wren_ref, params)
    if eltype(params) !== T || typeof(ref_lp) !== T || eltype(ref_grad) !== T
        @warn "Something was promoted away from $T" eltype(params) typeof(ref_lp) eltype(
            ref_grad,
        )
    end

    print_header()

    if RUN_CPU
        cpu_cases = [
            ("1. DPPL (y as arg)", :dppl, dppl_arg(X, LABELS, C)),
            ("2. DPPL (conditioned y)", :dppl, dppl_cond(X, C) | (; y=LABELS)),
            ("3. DPPL (@addlogprob!)", :dppl, dppl_addlogprob(X, LABELS, C)),
            ("4. Wren (no @batch)", :wren, wren_nobatch(X, LABELS, C)),
            ("5. Wren (@batch)", :wren, wren_batch(X, LABELS, C)),
            ("6. Wren (@addloglikelihood)", :wren, wren_addll(X, LABELS, C)),
        ]
        for (name, kind, model) in cpu_cases
            name = name * " [Enzyme]"
            try
                if kind === :dppl
                    obj = dppl_ldf(model)
                    f_lp = LogDensityProblems.logdensity
                    f_grad = LogDensityProblems.logdensity_and_gradient
                else
                    obj = VecModel(model, WREN_AD)
                    f_lp = lp
                    f_grad = lp_and_grad
                end
                v, g = f_grad(obj, params)
                check(name, v, g, ref_lp, ref_grad)
                print_row(name, bench(f_lp, obj, params), bench(f_grad, obj, params))
            catch e
                print_error(name, e)
            end
        end
    end

    reactant_cases = [
        ("5. Wren (@batch)", wren_batch(X, LABELS, C)),
        ("6. Wren (@addloglikelihood)", wren_addll(X, LABELS, C)),
    ]
    for (name, model) in reactant_cases
        name = name * " [Reactant]"
        try
            # This is where Reactant compiles the model, which can take a while.
            vm = VecModel(model, REACTANT_AD)
            v, g = lp_and_grad(vm, params)
            check(name, v, g, ref_lp, ref_grad)
            # Timings include sending the parameters to the device and the results back.
            print_row(name, bench(lp, vm, params), bench(lp_and_grad, vm, params))
        catch e
            print_error(name, e)
        end
    end
    return nothing
end

main()
