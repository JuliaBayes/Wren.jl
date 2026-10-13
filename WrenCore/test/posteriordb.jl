module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
using Distributions: Binomial, logpdf
using Enzyme
import ForwardDiff
using PosteriorDB: PosteriorDB
using StableRNGs
using Test
using WrenCore
using WrenDists: WrenDists

@testset "posteriordb.jl" begin
    @testset "$name" for name in available_posteriordb_models()
        model = posteriordb_model(name)
        @test isfinite(logjoint(model, rand(StableRNG(468), model)))

        vm = VecModel(model, DI.AutoForwardDiff())
        v = rand(StableRNG(468), vm)
        @test length(v) == vm.dimension
        logp, grad = lp_and_grad(vm, v)
        @test isfinite(logp)
        @test length(grad) == length(v)
        @test all(isfinite, grad)
    end

    @test_throws ArgumentError posteriordb_model("not-a-posterior")

    @testset "Rate posteriors: Stan density and unconstrained gradient" begin
        pdb = PosteriorDB.database()
        for (number, point) in (
            (2, [0.3, -0.4]), (2, [-1.2, 1.3]), (2, [2.0, -2.0]),
            (3, [0.3]), (3, [-1.2]), (3, [2.0]),
            (4, [0.3, -0.4]), (4, [-1.2, 1.3]), (4, [2.0, -2.0]),
            (5, [-0.4]), (5, [1.3]), (5, [-2.0]),
        )
            name = "Rate_$(number)_data-Rate_$(number)_model"
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == length(point)

            p = inv(1 + exp(-point[1]))
            jac = log(p) + log1p(-p)
            if number == 4
                q = inv(1 + exp(-point[2]))
                expected = logpdf(Binomial(data["n"], p), data["k"]) + jac +
                    log(q) + log1p(-q)
                expected_grad = [data["k"] + 1 - (data["n"] + 2) * p, 1 - 2q]
            elseif number == 2
                q = inv(1 + exp(-point[2]))
                expected = logpdf(Binomial(data["n1"], p), data["k1"]) + jac +
                    logpdf(Binomial(data["n2"], q), data["k2"]) + log(q) + log1p(-q)
                expected_grad = [
                    data["k1"] + 1 - (data["n1"] + 2) * p,
                    data["k2"] + 1 - (data["n2"] + 2) * q,
                ]
            else
                expected = logpdf(Binomial(data["n1"], p), data["k1"]) +
                    logpdf(Binomial(data["n2"], p), data["k2"]) + jac
                expected_grad = [
                    data["k1"] + data["k2"] + 1 -
                    (data["n1"] + data["n2"] + 2) * p,
                ]
            end
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, expected; atol=1e-9)
            @test isapprox(gradient, expected_grad; atol=1e-9)
        end
    end
end

end # module WrenPosteriorDBTests
