using WrenDists, Distributions, ForwardDiff, ReverseDiff, Mooncake, Enzyme, Test
using Random: Xoshiro
import DifferentiationInterface as DI

@testset "Sampling noise under AD: $T" for T in (Float32, Float64)
    x = T[0.7]
    z = randn(Xoshiro(8675309), T)
    zs = randn(Xoshiro(8675309), T, 8)
    es = -log.(rand(Xoshiro(8675309), T, 2, 4))
    cases = (
        ("Normal", s -> rand(Xoshiro(8675309), WrenDists.Normal(zero(only(s)), only(s))), z),
        ("IID Normal", s -> sum(rand(
            Xoshiro(8675309),
            WrenDists.Independent(WrenDists.Repeated(
                WrenDists.Normal(zero(only(s)), only(s)), (8,),
            )),
        )), sum(zs)),
        ("BatchedExponential", s -> sum(rand(
            Xoshiro(8675309), WrenDists.BatchedExponential(fill(only(s), 2, 4)),
        )), sum(es)),
    )
    backends = Base.get_extension(WrenDists, :WrenDistsTestUtils).TESTED_ADTYPES
    @testset "$name / $adname" for (name, f, expected) in cases, (adname, backend) in backends
        value, gradient = DI.value_and_gradient(f, backend, x)
        @test value ≈ f(x) rtol = 1e-5
        @test gradient ≈ [expected] rtol = 1e-5
    end
end

@testset "TDist shape sampling gradient" begin
    rng = Xoshiro(8675309)
    z, u = randn(rng), rand(rng)
    reference(v) = z / sqrt(Distributions.quantile(Distributions.Gamma(v / 2), u) / (v / 2))
    h = 1e-4
    expected = (reference(3.2 + h) - reference(3.2 - h)) / (2h)
    draw(v) = rand(Xoshiro(8675309), WrenDists.TDist(only(v), zero(only(v)), one(only(v))))
    backends = Base.get_extension(WrenDists, :WrenDistsTestUtils).TESTED_ADTYPES
    @testset "$name" for (name, backend) in backends
        @test DI.gradient(draw, backend, [3.2]) ≈ [expected] rtol = 1e-6
    end
end
