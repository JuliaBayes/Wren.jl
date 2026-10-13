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

    @testset "wells logistic posteriors: Stan density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "wells_data-wells_dist",
        )))
        y, dist, arsenic = data["switched"], data["dist"], data["arsenic"]
        dist100 = dist ./ 100.0
        c_dist100 = (dist .- sum(dist) / length(dist)) ./ 100.0
        c_arsenic = arsenic .- sum(arsenic) / length(arsenic)
        cases = (
            ("wells_data-wells_dist", hcat(ones(length(dist)), dist), false),
            ("wells_data-wells_dist100_model", hcat(dist100), true),
            ("wells_data-wells_interaction_model",
                hcat(dist100, arsenic, dist100 .* arsenic), true),
            ("wells_data-wells_interaction_c_model",
                hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic), true),
        )
        for (name, X, has_alpha) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            dimension = size(X, 2) + has_alpha
            @test vm.dimension == dimension
            for scale in (0.2, -0.3)
                point = scale .* sin.(1:dimension)
                logit_p = has_alpha ? point[1] .+ X * view(point, 2:dimension) : X * point
                p = @. inv(1 + exp(-logit_p))
                expected = sum(y[i] * logit_p[i] - log1p(exp(logit_p[i])) for i in eachindex(y))
                residual = y .- p
                expected_grad = has_alpha ? vcat(sum(residual), X' * residual) : X' * residual
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, expected; atol=1e-6)
                @test isapprox(gradient, expected_grad; atol=1e-6)
            end
        end
    end
end

end # module WrenPosteriorDBTests
