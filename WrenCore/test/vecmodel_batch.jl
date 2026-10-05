module WrenVecModelBatchTests

@info "testing vecmodel_batch.jl"

using ADTypes: AutoReactant
import DifferentiationInterface as DI
import ForwardDiff
import Enzyme
using Random
using StableRNGs
using Test
using WrenCore
using WrenDists: WrenDists as WD

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

@wren function linreg(X, y)
    N, D = size(X)
    @iid β ~ WD.Normal() size=D
    σ ~ WD.Exponential()
    μ = X * β
    @batch for i in 1:N
        @observe y[i] ~ WD.Normal(μ[i], σ)
    end
end

# Writes its parameters into the model argument `m`, so Enzyme needs `m` to be copied on
# every evaluation (see the "enzyme-copyargs" docs).
@wren function writes_arg(m, y)
    @batch for i in eachindex(y)
        m[i] ~ WD.Normal(0, 1)
        @observe y[i] ~ WD.Normal(m[i], 1)
    end
end

function make_model(rng)
    N, D = 20, 3
    X = randn(rng, N, D)
    y = X * [1.0, -2.0, 0.5] .+ 0.3 .* randn(rng, N)
    return linreg(X, y)
end

reference_lps(vm, P) = [lp(vm, p) for p in eachcol(P)]
function reference_grads(vm, P)
    return reduce(hcat, [ForwardDiff.gradient(Base.Fix1(lp, vm), p) for p in eachcol(P)])
end
function reference_hvps(vm, P, V)
    return reduce(
        hcat,
        [ForwardDiff.hessian(Base.Fix1(lp, vm), P[:, t]) * V[:, t] for t in axes(P, 2)],
    )
end

@testset "vecmodel_batch.jl" begin
    rng = StableRNG(468)
    model = make_model(rng)
    ref_vm = VecModel(model, AutoForwardDiff())
    D = ref_vm.dimension
    T = 5
    P = randn(rng, D, T)
    V = randn(rng, D, T)

    @testset "lp_batch and lp_and_grad_batch" begin
        for adtype in (AutoForwardDiff(), AutoEnzyme(; mode=Reverse))
            vm = VecModel(model, adtype)
            @test lp_batch(vm, P) ≈ reference_lps(ref_vm, P)
            lps, grads = lp_and_grad_batch(vm, P)
            @test lps ≈ reference_lps(ref_vm, P)
            @test size(grads) == size(P)
            @test grads ≈ reference_grads(ref_vm, P)
        end
    end

    @testset "dimension mismatch" begin
        @test_throws DimensionMismatch lp_batch(ref_vm, randn(D + 1, T))
        @test_throws DimensionMismatch lp_and_grad_batch(ref_vm, randn(D + 1, T))
    end

    @testset "no AD" begin
        vm = VecModel(model, nothing)
        @test lp_batch(vm, P) ≈ reference_lps(ref_vm, P)
        @test_throws ArgumentError lp_and_grad_batch(vm, P)
    end

    @testset "batched_lp_function is differentiable" begin
        f = batched_lp_function(VecModel(model, nothing))
        summed(x) = sum(f(x))
        @test f(P) ≈ reference_lps(ref_vm, P)
        @test DI.gradient(summed, AutoForwardDiff(), P) ≈ reference_grads(ref_vm, P)
        enzyme = AutoEnzyme(; mode=Reverse, function_annotation=Enzyme.Const)
        @test DI.gradient(summed, enzyme, P) ≈ reference_grads(ref_vm, P)
        hvp = only(
            DI.hvp(summed, DI.SecondOrder(AutoForwardDiff(), AutoForwardDiff()), P, (V,)),
        )
        @test hvp ≈ reference_hvps(ref_vm, P, V)
    end

    @testset "Enzyme copies model arguments" begin
        y = randn(rng, 4)
        vm = VecModel(writes_arg(zeros(4), y), AutoEnzyme(; mode=Reverse))
        Q = randn(rng, 4, T)
        # log N(m; 0, 1) + log N(y; m, 1) has gradient -m + (y - m).
        expected = -Q .+ (y .- Q)
        enzyme = AutoEnzyme(; mode=Reverse, function_annotation=Enzyme.Const)

        f = batched_lp_function(vm)
        @test DI.gradient(x -> sum(f(x)), enzyme, Q) ≈ expected
        g = logdensity_function(vm)
        @test DI.gradient(g, enzyme, Q[:, 1]) ≈ expected[:, 1]
        @test lp_batch(vm, Q) ≈ [lp(vm, q) for q in eachcol(Q)]
    end

    if RUN_REACTANT
        @testset "Reactant" begin
            vm = VecModel(model, AutoReactant(AutoEnzyme(; mode=Reverse)))

            @testset "lp_batch and lp_and_grad_batch" begin
                # A second batch size exercises the per-size compilation cache.
                for Q in (P, randn(rng, D, 2T))
                    @test lp_batch(vm, Q) ≈ reference_lps(ref_vm, Q)
                    lps, grads = lp_and_grad_batch(vm, Q)
                    @test lps ≈ reference_lps(ref_vm, Q)
                    @test grads ≈ reference_grads(ref_vm, Q)
                end
                @test length(vm.adprep.batch_primal) == 2
                @test length(vm.adprep.batch_value_and_gradient) == 2
            end

            @testset "batched_lp_function under @compile" begin
                f = batched_lp_function(vm)
                summed(x) = sum(f(x))
                grad(x) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(summed), x))
                hvp(x, v) = only(
                    Enzyme.autodiff(
                        Enzyme.Forward,
                        Enzyme.Const(grad),
                        Enzyme.Duplicated(x, v),
                    ),
                )

                # Function form of `@jit`: a macro here would fail to expand on platforms
                # where Reactant is not loaded.
                jit(g, args...) = Array(Reactant.compile(g, args)(args...))
                Pr, Vr = Reactant.to_rarray(P), Reactant.to_rarray(V)
                @test jit(f, Pr) ≈ reference_lps(ref_vm, P)
                @test jit(grad, Pr) ≈ reference_grads(ref_vm, P)
                @test jit(hvp, Pr, Vr) ≈ reference_hvps(ref_vm, P, V)
            end
        end
    end
end

end # module WrenVecModelBatchTests
