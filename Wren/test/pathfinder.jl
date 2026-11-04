module PathfinderTests

using Wren, Wren.Dists, StableRNGs, Test, Pathfinder
import Enzyme, ForwardDiff

@wren function positive_location(y)
    rate ~ LogNormal(0.7, 0.6)
    @observe y ~ Normal(log(rate), 1.0)
end

@testset "Pathfinder" begin
    model = positive_location(1.2)
    posterior_var = inv(inv(0.6^2) + 1)
    posterior_mean = posterior_var * (0.7 / 0.6^2 + 1.2)
    single = pathfinder(model; rng=StableRNG(42), ndraws=32, init=[-1.0])
    @test mean(single.fit_distribution)[1] ≈ posterior_mean atol=1e-5
    @test var(single.fit_distribution)[1] ≈ posterior_var atol=1e-5
    @test vec(Array(single.draws_transformed[@vn(rate)])) ≈ exp.(vec(single.draws))
    fit = multipathfinder(model, Int32(32); rng=StableRNG(42), init=[[-1.0]])
    path = only(fit.pathfinder_results)
    @test mean(path.fit_distribution)[1] ≈ posterior_mean atol=1e-5
    @test var(path.fit_distribution)[1] ≈ posterior_var atol=1e-5
    @test first(path.optim_trace.points) == [-1.0]
    @test size(path.draws, 2) == 32
    @test vec(Array(fit.draws_transformed[@vn(rate)])) ≈ exp.(vec(fit.draws))
    @test vec(Array(path.draws_transformed[@vn(rate)])) ≈ exp.(vec(path.draws))
    repeated = multipathfinder(model, 32; rng=StableRNG(42), init=[[-1.0]])
    @test repeated.draws == fit.draws
    counted = multipathfinder(model, 12; rng=StableRNG(42), nruns=Int32(1))
    @test length(counted.pathfinder_results) == 1

    multi = multipathfinder(
        model,
        12;
        rng=StableRNG(0x6179616E616D69),
        adtype=AutoForwardDiff(),
        init=[[-1.0], [2.0]],
        ndraws_per_run=32,
    )
    @test vec(Array(multi.draws_transformed[@vn(rate)])) ≈ exp.(vec(multi.draws))
    for (path, initial) in zip(multi.pathfinder_results, ([-1.0], [2.0]))
        @test mean(path.fit_distribution)[1] ≈ posterior_mean atol=1e-5
        @test first(path.optim_trace.points) == initial
        @test vec(Array(path.draws_transformed[@vn(rate)])) ≈ exp.(vec(path.draws))
    end
end

end
