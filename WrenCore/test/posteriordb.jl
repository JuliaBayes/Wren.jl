module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
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

    @testset "kidscore linear posteriors: density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "kidiq-kidscore_momhs",
        )))
        y, mom_hs, mom_iq = data["kid_score"], data["mom_hs"], data["mom_iq"]
        intercept = ones(length(y))
        cases = (
            ("kidiq-kidscore_momhs", hcat(intercept, mom_hs)),
            ("kidiq-kidscore_momiq", hcat(intercept, mom_iq)),
            ("kidiq-kidscore_momhsiq", hcat(intercept, mom_hs, mom_iq)),
        )
        for (name, X) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            k = size(X, 2)
            @test vm.dimension == k + 1
            for log_sigma in (log(50.0), log(85.0))
                beta = 0.2 .* cos.(1:k)
                point = vcat(beta, log_sigma)
                sigma = exp(log_sigma)
                residual = y - X * beta
                expected = -length(y) * log(sigma * sqrt(2π)) -
                    sum(abs2, residual) / (2sigma^2) +
                    log(2 / (π * 2.5)) - log1p((sigma / 2.5)^2) + log_sigma
                expected_grad = vcat(
                    X' * residual / sigma^2,
                    -length(y) + sum(abs2, residual) / sigma^2 -
                        2sigma^2 / (2.5^2 + sigma^2) + 1,
                )
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, expected; atol=1e-6)
                @test isapprox(gradient, expected_grad; atol=1e-6)
            end
        end
    end
end

end # module WrenPosteriorDBTests
