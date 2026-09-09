using LinearAlgebra
using Plaice
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff

# Test utilities for Plaice.test_all
Plaice.is_continuous(::WrenDists.UnivariateDist) = true
Plaice.is_continuous(::WrenDists.DiscreteUnivariateDist) = false
Plaice.is_continuous(::WrenDists.MultivariateDist) = true
Plaice.is_continuous(::WrenDists.DiscreteMultivariateDist) = false
Plaice.test_name(d::WrenDists.WrenDist) = repr(d)
Plaice.rand_safe_ad(d::WrenDists.WrenDist) = rand(d)
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

# Dirichlet needs special logjac test helpers because vec_length != unconstrained_vec_length
Plaice.to_vec_for_logjac_test(::WrenDists.Dirichlet) = x -> x[1:(end-1)]
Plaice.from_vec_for_logjac_test(::WrenDists.Dirichlet) = y -> vcat(y, 1 - sum(y))
