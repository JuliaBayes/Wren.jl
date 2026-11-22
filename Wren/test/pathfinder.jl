module PathfinderTests

using Wren, Wren.Dists, StableRNGs, Test, Pathfinder
import Enzyme, ForwardDiff

@wren function positive_location(y)
    rate ~ LogNormal(0.7, 0.6)
    @observe y ~ Normal(log(rate), 1.0)
end

# The log density is -Inf when |x| > 5, so optimisation fails if it starts there.
@wren function bounded_location(y)
    x ~ Normal(0.0, 1.0)
    @observe y ~ Uniform(x - 5, x + 5)
end

@testset "Pathfinder" begin
    model = positive_location(1.2)
    posterior_var = inv(inv(0.6^2) + 1)
    posterior_mean = posterior_var * (0.7 / 0.6^2 + 1.2)
    # Starting points in model space; their unconstrained values are log(rate).
    start(rate) = FromValues(VarNamedTuple(; rate))
    single = pathfinder(model; rng=StableRNG(42), ndraws=32, init=start(exp(-1.0)))
    @test mean(single.fit_distribution)[1] ≈ posterior_mean atol=1e-5
    @test var(single.fit_distribution)[1] ≈ posterior_var atol=1e-5
    @test first(single.optim_trace.points) ≈ [-1.0]
    @test vec(Array(single.draws_transformed[@vn(rate)])) ≈ exp.(vec(single.draws))
    fit = multipathfinder(model, Int32(32); rng=StableRNG(42), init=[start(exp(-1.0))])
    path = only(fit.pathfinder_results)
    @test mean(path.fit_distribution)[1] ≈ posterior_mean atol=1e-5
    @test var(path.fit_distribution)[1] ≈ posterior_var atol=1e-5
    @test first(path.optim_trace.points) ≈ [-1.0]
    @test size(fit.draws, 2) == 32
    @test vec(Array(fit.draws_transformed[@vn(rate)])) ≈ exp.(vec(fit.draws))
    @test vec(Array(path.draws_transformed[@vn(rate)])) ≈ exp.(vec(path.draws))
    repeated = multipathfinder(model, 32; rng=StableRNG(42), init=[start(exp(-1.0))])
    @test repeated.draws == fit.draws
    counted = multipathfinder(model, 12; rng=StableRNG(42), nruns=Int32(1))
    @test length(counted.pathfinder_results) == 1
    @test size(only(counted.pathfinder_results).draws, 2) == 12

    # With no `init`, starting points are drawn uniformly from [-2, 2] in unconstrained space.
    default_single = pathfinder(model; rng=StableRNG(1), ndraws=8)
    @test mean(default_single.fit_distribution)[1] ≈ posterior_mean atol=1e-5
    @test all(abs.(first(default_single.optim_trace.points)) .<= 2)

    @testset "rejected keywords" begin
        @test_throws ArgumentError pathfinder(model; init=[-1.0])
        @test_throws ArgumentError multipathfinder(model, 8; init=[[-1.0]])
        @test_throws ArgumentError multipathfinder(model, 8; init=[-1.0])
        @test_throws ArgumentError pathfinder(model; init_scale=0)
        @test_throws ArgumentError pathfinder(model; init_scale=1, init_sampler=FromPrior())
        @test_throws ArgumentError pathfinder(model; latent_source=FromPrior())
    end

    multi = multipathfinder(
        model,
        12;
        rng=StableRNG(0x6179616E616D69),
        adtype=AutoForwardDiff(),
        init=[start(exp(-1.0)), start(exp(2.0))],
        ndraws_per_run=32,
    )
    @test vec(Array(multi.draws_transformed[@vn(rate)])) ≈ exp.(vec(multi.draws))
    for (path, initial) in zip(multi.pathfinder_results, ([-1.0], [2.0]))
        @test mean(path.fit_distribution)[1] ≈ posterior_mean atol=1e-5
        @test first(path.optim_trace.points) ≈ initial
        @test vec(Array(path.draws_transformed[@vn(rate)])) ≈ exp.(vec(path.draws))
    end

    @testset "init_sampler is used for retries" begin
        bmodel = bounded_location(0.0)
        bad = FromValues(VarNamedTuple(; x=10.0))
        good = FromValues(VarNamedTuple(; x=0.5))
        res = pathfinder(bmodel; rng=StableRNG(1), init=bad, init_sampler=good, ndraws=8)
        @test res.num_tries == 2
        @test first(res.optim_trace.points) ≈ [0.5]
        mres = multipathfinder(
            bmodel,
            8;
            rng=StableRNG(1),
            init=[bad, good],
            init_sampler=good,
        )
        @test [p.num_tries for p in mres.pathfinder_results] == [2, 1]
    end

    @testset "init_scale sets the starting point when init is not given" begin
        scale = 0.25
        res = pathfinder(model; rng=StableRNG(1), init_scale=scale, ndraws=8)
        @test all(abs.(first(res.optim_trace.points)) .<= scale)
        mres = multipathfinder(model, 8; rng=StableRNG(1), init_scale=scale)
        for p in mres.pathfinder_results
            @test all(abs.(first(p.optim_trace.points)) .<= scale)
        end
    end

    @testset "init_sampler sets the starting point when init is not given" begin
        good = FromValues(VarNamedTuple(; rate=exp(0.5)))
        res = pathfinder(model; rng=StableRNG(1), init_sampler=good, ndraws=8)
        @test first(res.optim_trace.points) ≈ [0.5]
    end

    @testset "init_scale sets the retry sampler" begin
        bmodel = bounded_location(0.0)
        bad = FromValues(VarNamedTuple(; x=10.0))
        scale = 0.25
        res = pathfinder(bmodel; rng=StableRNG(1), init=bad, init_scale=scale, ndraws=8)
        @test res.num_tries >= 2
        @test all(abs.(first(res.optim_trace.points)) .<= scale)
        mres =
            multipathfinder(bmodel, 8; rng=StableRNG(1), init=[bad, bad], init_scale=scale)
        for p in mres.pathfinder_results
            @test p.num_tries >= 2
            @test all(abs.(first(p.optim_trace.points)) .<= scale)
        end
    end
end

end
