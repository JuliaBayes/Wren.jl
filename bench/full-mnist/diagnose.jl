# Finds out which part of the MNIST model is slow under Reactant on a GPU. See HANDOFF.md
# for the background.
#
# This times each piece of the model on its own under Reactant, both the value and the
# gradient with respect to W and b:
#
#   - `W * X'` (as in the model) vs `W * Xt`, where `Xt = permutedims(X)` is made in
#     advance so that no transpose is needed
#   - then adding `.+ b`, the `vcat`, and the likelihood, one at a time
#   - a hand-written log density for the whole model vs Wren's `VecModel`, which shows
#     whether Wren adds anything (for example the copy of the model arguments on each call)
#
# It also reports how long each piece takes to compile, and at the end profiles the GPU
# kernels for the full likelihood, with `X'` and with `Xt`.
#
# Run from this folder with:
#
#     julia --project=. diagnose.jl | tee diagnose.txt
#
# Environment variables:
#
#   - WREN_BENCH_SECONDS: Chairmarks sampling time per measurement (default 2.0)
#   - WREN_DIAG_N: number of images to use (default: all 60,000). A small number is
#     useful for checking that the script works before paying for GPU time.
#   - WREN_DIAG_TYPES: float types for the pieces (default `Float64,Float32`). The Wren
#     `VecModel` rows always use WrenCore's log-probability type preference instead.
#   - WREN_DIAG_UNTRACED: set to `true` to also time `W * X'` with `X` as a constant that
#     Reactant doesn't trace (default `false`). This builds the whole of `X` into the
#     compiled program, which may take a very long time to compile, so it runs last.
#   - WREN_DIAG_PROFILE: set to `false` to skip the kernel profiles (default `true`)

using ADTypes: AutoEnzyme, AutoReactant
using Chairmarks: @be
using Enzyme
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

const SECONDS = parse(Float64, get(ENV, "WREN_BENCH_SECONDS", "2.0"))
const TYPES = map(
    s -> Dict("Float64" => Float64, "Float32" => Float32)[strip(s)],
    split(get(ENV, "WREN_DIAG_TYPES", "Float64,Float32"), ','),
)
const RUN_UNTRACED = parse(Bool, get(ENV, "WREN_DIAG_UNTRACED", "false"))
const RUN_PROFILE = parse(Bool, get(ENV, "WREN_DIAG_PROFILE", "true"))

const mnist_data = MNIST(split=:train)
const N = parse(Int, get(ENV, "WREN_DIAG_N", string(length(mnist_data.targets))))
# N x 784 matrix of pixel values in [0, 1], one row per image
const X64 = Matrix(Float64.(reshape(mnist_data.features, 28 * 28, :))')[1:N, :]
# Convert labels from 0-9 to 1-10 for 1-based indexing in Julia
const LABELS = (mnist_data.targets .+ 1)[1:N]
const C = 10 # Number of labels
const D = size(X64, 2)

# ---------------------------------------------------------------------------------------
# The pieces. Each returns a scalar, so that the gradient is well defined.

# Vectorised form of sum_i log(softmax(logits[:, i])[y[i]])
# This uses linear indices because Reactant can't trace `CartesianIndex.(y, 1:N)`.
function categorical_loglik(logits, y)
    K, N = size(logits)
    return sum(logits[y .+ K .* (0:(N-1))]) - sum(logsumexp(logits; dims=1))
end
add_reference_row(nonreference_logits) = vcat(
    zeros(eltype(nonreference_logits), 1, size(nonreference_logits, 2)),
    nonreference_logits,
)

matmul_adjoint(W, b, X, Xt, y) = sum(W * X')
matmul_pretransposed(W, b, X, Xt, y) = sum(W * Xt)
plus_b(W, b, X, Xt, y) = sum(W * Xt .+ b)
plus_vcat(W, b, X, Xt, y) = sum(add_reference_row(W * Xt .+ b))
plus_softmax(W, b, X, Xt, y) = sum(softmax(add_reference_row(W * Xt .+ b); dims=1))
loglik_adjoint(W, b, X, Xt, y) = categorical_loglik(add_reference_row(W * X' .+ b), y)
loglik_pretransposed(W, b, X, Xt, y) = categorical_loglik(add_reference_row(W * Xt .+ b), y)

const PIECES = [
    ("sum(W * X')", matmul_adjoint),
    ("sum(W * Xt)", matmul_pretransposed),
    ("... .+ b", plus_b),
    ("... + vcat", plus_vcat),
    ("... + softmax", plus_softmax),
    ("loglik, X'", loglik_adjoint),
    ("loglik, Xt", loglik_pretransposed),
]

# The gradient with respect to W and b, keeping the data constant.
function value_and_grad(f, W, b, X, Xt, y)
    result = Enzyme.gradient(
        Enzyme.ReverseWithPrimal,
        Const(f),
        W,
        b,
        Const(X),
        Const(Xt),
        Const(y),
    )
    return result.val, result.derivs[1], result.derivs[2]
end

# The whole log density, written by hand, taking one parameter vector like a `VecModel`
# does. The order of the parameters doesn't need to match Wren's, since this is only
# timed, not compared with Wren.
function handwritten_lp(params, X, Xt, y, use_adjoint)
    W = reshape(params[1:((C-1)*D)], C - 1, D)
    b = params[((C-1)*D+1):end]
    T = eltype(params)
    logprior = -sum(abs2, params) / 2 - length(params) * T(log(2π)) / 2
    f = use_adjoint ? loglik_adjoint : loglik_pretransposed
    return logprior + f(W, b, X, Xt, y)
end
function handwritten_value_and_grad(params, X, Xt, y, use_adjoint)
    result = Enzyme.gradient(
        Enzyme.ReverseWithPrimal,
        Const(handwritten_lp),
        params,
        Const(X),
        Const(Xt),
        Const(y),
        Const(use_adjoint),
    )
    return result.val, result.derivs[1]
end

# ---------------------------------------------------------------------------------------
# Wren versions of variant 6 from `gpu.jl`, with `X'` and with `Xt`.

@wren function wren_addll(X, y, C)
    N, D = size(X)
    @iid W ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1, D)
    @iid b ~ WD.Normal(zero(eltype(X)), one(eltype(X))) size=(C-1)
    @addloglikelihood categorical_loglik(add_reference_row(W * X' .+ b), y)
end

@wren function wren_addll_pretransposed(Xt, y, C)
    D, N = size(Xt)
    @iid W ~ WD.Normal(zero(eltype(Xt)), one(eltype(Xt))) size=(C-1, D)
    @iid b ~ WD.Normal(zero(eltype(Xt)), one(eltype(Xt))) size=(C-1)
    @addloglikelihood categorical_loglik(add_reference_row(W * Xt .+ b), y)
end

# ---------------------------------------------------------------------------------------
# Timing helpers

# Copy results back to the host, so that the timings include waiting for the GPU to finish.
tohost(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)
tohost(x::Reactant.ConcreteRArray) = Array(x)
tohost(x::Tuple) = map(tohost, x)
tohost(x) = x

function time_call(f, args...)
    tohost(f(args...)) # warm up
    return median(@be tohost($f($args...)) seconds=SECONDS).time
end

function print_header()
    @printf(
        "%-34s%-9s%12s%12s%12s%12s\n",
        "piece",
        "type",
        "compile",
        "value",
        "compile",
        "grad"
    )
    @printf("%-34s%-9s%12s%12s%12s%12s\n", "", "", "(value)", "", "(grad)", "")
    println(repeat('─', 91))
end

function print_row(name, T, compile_value, value, compile_grad, grad)
    @printf(
        "%-34s%-9s%10.1f s%9.3f ms%10.1f s%9.3f ms\n",
        name,
        string(T),
        compile_value,
        value * 1e3,
        compile_grad,
        grad * 1e3,
    )
end

print_error(name, T, e) = @printf(
    "%-34s%-9sERROR: %s\n",
    name,
    string(T),
    first(split(sprint(showerror, e), '\n'))
)

function run_piece(name, f, T, W, b, X, Xt, y)
    try
        t_cv = @elapsed fv = @compile f(W, b, X, Xt, y)
        tv = time_call(fv, W, b, X, Xt, y)
        t_cg = @elapsed fg = @compile value_and_grad(f, W, b, X, Xt, y)
        tg = time_call(fg, f, W, b, X, Xt, y)
        print_row(name, T, t_cv, tv, t_cg, tg)
    catch e
        print_error(name, T, e)
    end
end

function run_handwritten(name, T, params, X, Xt, y, use_adjoint)
    try
        t_cv = @elapsed fv = @compile handwritten_lp(params, X, Xt, y, use_adjoint)
        tv = time_call(fv, params, X, Xt, y, use_adjoint)
        t_cg =
            @elapsed fg = @compile handwritten_value_and_grad(params, X, Xt, y, use_adjoint)
        tg = time_call(fg, params, X, Xt, y, use_adjoint)
        print_row(name, T, t_cv, tv, t_cg, tg)
    catch e
        print_error(name, T, e)
    end
end

# `VecModel` compiles both the value and the gradient when it is created, so only the total
# compile time is known. It is shown in the "compile (value)" column.
function run_vecmodel(name, model)
    T = WrenCore.LogProbType
    try
        t_c = @elapsed vm = VecModel(model, AutoReactant(AutoEnzyme(; mode=Reverse)))
        params = vm.sample
        tv = time_call(lp, vm, params)
        tg = time_call(lp_and_grad, vm, params)
        print_row(name, T, t_c, tv, NaN, tg)
    catch e
        print_error(name, T, e)
    end
end

# `X` as a constant that Reactant doesn't trace, following the example in
# https://github.com/EnzymeAD/Reactant.jl/issues/3167
struct UntracedMatmul{A}
    X::A
end
(m::UntracedMatmul)(W) = sum(W * m.X')

function run_untraced(T, W, X)
    m = UntracedMatmul(X)
    try
        t_cv = @elapsed fv = @compile m(W)
        tv = time_call(fv, W)
        print_row("sum(W * X'), X untraced", T, t_cv, tv, NaN, NaN)
    catch e
        print_error("sum(W * X'), X untraced", T, e)
    end
end

# ---------------------------------------------------------------------------------------

function main()
    println("Reactant device: ", string(first(Reactant.devices())))
    println("Wren log-probability type: ", WrenCore.LogProbType)
    println("N = $N images, D = $D pixels")
    println("Chairmarks sampling time per measurement: $(SECONDS) s\n")
    print_header()

    rng = Xoshiro(468)
    device_data = Dict()
    for T in TYPES
        X = Reactant.to_rarray(T.(X64))
        Xt = Reactant.to_rarray(permutedims(T.(X64)))
        y = Reactant.to_rarray(LABELS)
        W = Reactant.to_rarray(randn(rng, T, C - 1, D))
        b = Reactant.to_rarray(randn(rng, T, C - 1))
        params = Reactant.to_rarray(randn(rng, T, (C - 1) * (D + 1)))
        device_data[T] = (; X, Xt, y, W, b)

        for (name, f) in PIECES
            run_piece(name, f, T, W, b, X, Xt, y)
        end
        run_handwritten("hand-written lp, X'", T, params, X, Xt, y, true)
        run_handwritten("hand-written lp, Xt", T, params, X, Xt, y, false)
    end

    # These use the data on the host. VecModel moves it to the device itself.
    T = WrenCore.LogProbType
    run_vecmodel("Wren VecModel, X'", wren_addll(T.(X64), LABELS, C))
    run_vecmodel(
        "Wren VecModel, Xt",
        wren_addll_pretransposed(permutedims(T.(X64)), LABELS, C),
    )

    if RUN_UNTRACED
        for T in TYPES
            run_untraced(T, device_data[T].W, T.(X64))
        end
    end
    println("\n(NaN means not measured.)")

    if RUN_PROFILE
        for T in TYPES
            (; X, Xt, y, W, b) = device_data[T]
            for (name, f) in
                (("loglik, X'", loglik_adjoint), ("loglik, Xt", loglik_pretransposed))
                println("\n", "="^91)
                println("Kernel profile: gradient of $name, $T")
                println("="^91)
                try
                    fg = @compile value_and_grad(f, W, b, X, Xt, y)
                    Reactant.Profiler.@profile fg(f, W, b, X, Xt, y)
                catch e
                    println("ERROR: ", first(split(sprint(showerror, e), '\n')))
                end
            end
        end
    end
    return nothing
end

main()
