using Plaice
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff

# Test utilities for Plaice.test_all
Plaice.is_continuous(::WrenDists.UnivariateDist) = true
Plaice.is_continuous(::WrenDists.DiscreteUnivariateDist) = false
Plaice.test_name(d::WrenDists.UnivariateDist) = repr(d)
Plaice.rand_safe_ad(d::WrenDists.UnivariateDist) = rand(d)
Plaice.can_test_in_support(::WrenDists.UnivariateDist, x) = true
function Plaice.test_in_support(::WrenDists.UnconstrainedUnivariateDist, x)
    @test isfinite(x)
end
function Plaice.test_in_support(::WrenDists.PositiveUnivariateDist, x)
    @test x > 0
end
function Plaice.test_in_support(::WrenDists.Beta, x)
    @test 0 <= x <= 1
end
function Plaice.test_in_support(d::WrenDists.Uniform, x)
    @test d.a <= x <= d.b
end

univariates = [
    # Continuous unconstrained
    WrenDists.Normal(),
    WrenDists.Cauchy(),
    WrenDists.Laplace(),
    # Continuous positive
    WrenDists.Exponential(),
    WrenDists.LogNormal(),
    WrenDists.Weibull(),
    WrenDists.Gamma(),
    # Continuous bounded
    WrenDists.Beta(),
    WrenDists.Uniform(),
    # Discrete
    WrenDists.Bernoulli(),
    WrenDists.Binomial(5, 0.5),
    WrenDists.Categorical([0.2, 0.5, 0.3]),
    WrenDists.Geometric(),
    WrenDists.Poisson(3.0),
]

@testset "Plaice univariates" begin
    for d in univariates
        Plaice.test_all(d; expected_zero_allocs=(from_vec, from_unconstrained_vec))
    end
end
