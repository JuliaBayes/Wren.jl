module WrenCoreOptimizationTests

using WrenCore, Distributions, Optimization, OptimizationOptimJL, Test
import Enzyme

@wren function poisson_rate(count)
    rate ~ Gamma(2.0, 1.0)
    @observe count ~ Poisson(rate)
end

@testset "MAP and MLE" begin
    initial = FromValues(VarNamedTuple(rate=1.5))
    map_fit = MAP(poisson_rate(3), LBFGS(); init=initial, abstol=1e-10)
    mle_fit = MLE(poisson_rate(3), LBFGS(); init=initial, abstol=1e-10)
    @test map_fit.parameters[@vn(rate)] ≈ 2.0 rtol=1e-6
    @test mle_fit.parameters[@vn(rate)] ≈ 3.0 rtol=1e-6
    @test Optimization.SciMLBase.successful_retcode(map_fit.solution)
    @test Optimization.SciMLBase.successful_retcode(mle_fit.solution)
    @test map_fit isa PointEstimate
    @test parameters(map_fit) === map_fit.parameters
    @test map_fit.lp ≈ logjoint(poisson_rate(3), map_fit.parameters)
    @test mle_fit.lp ≈ loglikelihood(poisson_rate(3), mle_fit.parameters)
    @test map_fit.lp ≈ lp(map_fit.vecmodel, map_fit.solution.u)
    @test occursin("PointEstimate", sprint(show, MIME"text/plain"(), map_fit))
    # `init` also accepts a plain VarNamedTuple.
    vnt_fit = MAP(poisson_rate(3), LBFGS(); init=VarNamedTuple(rate=1.5), abstol=1e-10)
    @test vnt_fit.parameters[@vn(rate)] ≈ 2.0 rtol=1e-6
    @test_throws ArgumentError MAP(poisson_rate(3), LBFGS(); u0=[0.0])
end

@testset "initialisation" begin
    # `init` also accepts an AbstractInitialization, as for `nuts`.
    init = FindFiniteDensity(VarNamedTuple(rate=1.5))
    fit = MAP(poisson_rate(3), LBFGS(); init, abstol=1e-10)
    @test fit.parameters[@vn(rate)] ≈ 2.0 rtol=1e-6

    # A starting point with non-finite density is rejected.
    @wren function finite_region()
        x ~ Normal()
        @addloglikelihood (x > 1 ? 0.0 : -Inf)
    end
    @test_throws ArgumentError MAP(finite_region(), LBFGS(); init=VarNamedTuple(x=0.0))
    fit = MAP(finite_region(), LBFGS(); init=VarNamedTuple(x=2.0))
    @test fit.parameters[@vn(x)] > 1
end

end
