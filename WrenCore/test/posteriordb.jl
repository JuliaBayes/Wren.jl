module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
using Distributions: Cauchy, Normal, logpdf
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

    @testset "noncentered schools: Stan density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "eight_schools-eight_schools_noncentered",
        )))
        J, y, sigma = data["J"], data["y"], data["sigma"]
        vm = VecModel(posteriordb_model("eight_schools-eight_schools_noncentered"),
            AutoEnzyme(; mode=Reverse))
        @test vm.dimension == J + 2

        function reference(v)
            theta_trans, mu, logtau = view(v, 1:J), v[J+1], v[J+2]
            tau = exp(logtau)
            theta = theta_trans .* tau .+ mu
            return sum(logpdf(Normal(), x) for x in theta_trans) +
                logpdf(Normal(0, 5), mu) + logpdf(Cauchy(0, 5), tau) + log(2) +
                logtau + sum(logpdf(Normal(theta[i], sigma[i]), y[i]) for i in eachindex(y))
        end
        for point in (
            vcat(collect(range(-0.8, 0.8; length=J)), 0.4, -0.3),
            vcat(sin.(1:J), -1.1, 0.5),
        )
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-9)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-8)
        end
    end

    @testset "pooled radon: Stan density and gradient" begin
        pdb = PosteriorDB.database()
        for name in ("radon_all-radon_pooled", "radon_mn-radon_pooled")
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
            y, floor_measure = data["log_radon"], data["floor_measure"]
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == 3

            function reference(v)
                alpha, beta, logsigma = v
                sigma_y = exp(logsigma)
                return logpdf(Normal(0, 10), alpha) + logpdf(Normal(0, 10), beta) +
                    logpdf(Normal(), sigma_y) + log(2) + logsigma +
                    sum(logpdf(Normal(alpha + beta * floor_measure[i], sigma_y), y[i])
                        for i in eachindex(y))
            end
            for point in ([0.2, -0.5, log(0.8)], [1.1, 0.4, log(1.4)])
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-6)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-6)
            end
        end
    end
end

end # module WrenPosteriorDBTests
