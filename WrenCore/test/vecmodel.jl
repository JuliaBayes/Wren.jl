module WrenVecModelTests

@info "testing vecmodel.jl"

import DifferentiationInterface as DI
using Distributions
import ForwardDiff
import ReverseDiff
import Mooncake
import Enzyme
using LinearAlgebra
import LogDensityProblems as LDP
using Plaice
using Random
using StableRNGs
using Test
using WrenCore

const TESTED_ADTYPES = (
    "ForwardDiff" => DI.AutoForwardDiff(),
    "ReverseDiff" => DI.AutoReverseDiff(),
    "ReverseDiffCpl" => DI.AutoReverseDiff(; compile=true),
    "MooncakeFwd" => DI.AutoMooncakeForward(),
    "MooncakeRvs" => DI.AutoMooncake(),
    "EnzymeFwd" => DI.AutoEnzyme(; mode=Enzyme.Forward),
    "EnzymeRvs" => DI.AutoEnzyme(; mode=Enzyme.Reverse),
)

function reference_lp(dist, y::AbstractVector)
    tfm = Plaice.from_unconstrained_vec(dist)
    x, inv_logjac = Plaice.with_logabsdet_jacobian(tfm, y)
    logp = logdensityof(dist, x)
    return logp + inv_logjac
end
function reference_grad(dist, y::AbstractVector)
    # Would be nice to use finite differences but it's too inaccurate
    DI.gradient(Base.Fix1(reference_lp, dist), DI.AutoForwardDiff(), y)
end

@testset "vecmodel.jl" begin
    @testset "VecModel constructors" begin
        @testset "defaults + fields" begin
            @wren function f()
                x ~ Beta(2, 2)
            end
            vm = VecModel(f(), nothing)
            @test vm.parameter_space isa UnconstrainedSpace
            @test vm.lp_function == logjoint_transformed
            @test vm.sample isa AbstractVector{<:Real}
            @test length(vm.sample) == Plaice.unconstrained_vec_length(Beta(2, 2))
            @test vm.dimension == Plaice.unconstrained_vec_length(Beta(2, 2))
        end
    end

    @testset "correctness on a few distributions" begin
        @testset "$(nameof(typeof(d)))" for d in (
            Normal(),
            Beta(2, 2),
            MvNormal(zeros(3), I),
            LKJCholesky(3, 0.5),
        )
            @wren function f(d)
                x ~ d
            end
            grad_rtol = 1e-7

            @testset "$name" for (name, adtype) in TESTED_ADTYPES
                if d isa Distributions.AbstractMvNormal && adtype isa AutoEnzyme
                    continue
                    # Reverse mode:
                    # https://github.com/EnzymeAD/Enzyme.jl/issues/3527
                    # https://github.com/EnzymeAD/Enzyme.jl/issues/3528
                    #
                    # Forward mode: MvNormal runs fine but then causes LKJCholesky to fail
                    # https://github.com/EnzymeAD/Enzyme.jl/issues/3529
                end
                @info nameof(typeof(d)), name

                vm = VecModel(f(d), adtype)
                x = rand(vm)
                N = length(x)
                @test N == Plaice.unconstrained_vec_length(d)

                @test LDP.capabilities(typeof(vm)) == LDP.LogDensityOrder{1}()
                @test LDP.dimension(vm) == N

                for _ in 1:10
                    # check with a few random vectors to make sure it's unconstrained
                    y = randn(N) .* 5
                    lp_expected = reference_lp(d, y)
                    grad_expected = reference_grad(d, y)

                    @test lp(vm, y) ≈ lp_expected

                    lp2, grad = lp_and_grad(vm, y)
                    @test lp2 ≈ lp_expected
                    @test grad ≈ grad_expected rtol=grad_rtol

                    similar_grad = zeros(N)
                    lp3, grad2 = lp_and_grad!(similar_grad, vm, y)
                    @test lp3 ≈ lp_expected
                    @test grad2 ≈ grad_expected rtol=grad_rtol
                    @test similar_grad ≈ grad_expected rtol=grad_rtol

                    @test LDP.logdensity(vm, y) ≈ lp_expected
                    lp4, grad3 = LDP.logdensity_and_gradient(vm, y)
                    @test lp4 ≈ lp_expected
                    @test grad3 ≈ grad_expected rtol=grad_rtol
                end
            end
        end
    end

    @testset "construction without ad" begin
        @wren function f()
            x ~ Beta(2, 2)
        end
        model = f()

        for noad in (nothing, DI.ADTypes.NoAutoDiff())
            vm = VecModel(model, noad)
            yvec = rand(vm)
            # Should still be able to calculate logp
            @test LDP.capabilities(typeof(vm)) == LDP.LogDensityOrder{0}()
            @test LDP.dimension(vm) == vm.dimension
            @test LDP.logdensity(vm, yvec) ≈ reference_lp(Beta(2, 2), yvec)

            # but not gradients
            @test_throws Exception LDP.logdensity_and_gradient(vm, yvec)
            @test_throws Exception lp_and_grad(vm, yvec)
            @test_throws Exception lp_and_grad!(similar(yvec), vm, yvec)
        end
    end

    @testset "vectorise_latents and rand" begin
        @wren function fexp()
            x ~ Exponential()
        end
        model = fexp()

        vm = VecModel(model, nothing)

        x = @inferred rand(vm)
        @test x isa AbstractVector{<:Real}
        @test length(x) == 1

        # check reproducibility
        @test rand(Xoshiro(468), vm) == rand(Xoshiro(468), vm)

        # check that there are some negative values, i.e., it's unconstrained space
        xs = vcat([rand(StableRNG(i), vm) for i in 1:100]...)
        @test any(xs .< 0)

        for X in [0.1, 1.0, 10.0]
            prms = FromValues(VarNamedTuple(x=X))
            x = @inferred vectorise_latents(prms, vm)
            @test x ≈ [log(X)]
        end
    end
end

end # module WrenVecModelTests
