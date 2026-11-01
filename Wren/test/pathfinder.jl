module PathfinderTests

using Wren, Wren.Dists, StableRNGs, Test
import Enzyme, ForwardDiff, Pathfinder

@wren function positive_location(y)
    rate ~ LogNormal(0.7, 0.6)
    @observe y ~ Normal(log(rate), 1.0)
end

@testset "Pathfinder" begin
    model = positive_location(1.2)
    posterior_var = inv(inv(0.6^2) + 1)
    posterior_mean = posterior_var * (0.7 / 0.6^2 + 1.2)
    fit = pathfinder(StableRNG(42), model, Int32(32); init=[[-1.0]])
    path = only(fit.pathfinder_results)
    @test mean(path.fit_distribution)[1] ≈ posterior_mean atol=1e-5
    @test var(path.fit_distribution)[1] ≈ posterior_var atol=1e-5
    @test first(path.optim_trace.points) == [-1.0]
    @test size(path.draws, 2) == 32
    @test vec(Array(fit.draws_transformed[@vn(rate)])) ≈ exp.(vec(fit.draws))
    repeated = pathfinder(StableRNG(42), model, 32; init=[[-1.0]])
    @test repeated.draws == fit.draws

    multi = pathfinder(
        model,
        12;
        adtype=AutoForwardDiff(),
        init=[[-1.0], [2.0]],
        ndraws_per_run=32,
    )
    @test vec(Array(multi.draws_transformed[@vn(rate)])) ≈ exp.(vec(multi.draws))
    for (path, initial) in zip(multi.pathfinder_results, ([-1.0], [2.0]))
        @test mean(path.fit_distribution)[1] ≈ posterior_mean atol=1e-5
        @test first(path.optim_trace.points) == initial
    end
end

end
