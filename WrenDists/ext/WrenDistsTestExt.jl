module WrenDistsTestExt

using WrenDists: WrenDists, logdensityof, ADTestCase
import DifferentiationInterface as DI
using Distributions: logpdf
using EnzymeCore: EnzymeCore
using Plaice: Plaice
using Random: Xoshiro
using Test

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

# Independent / IID
Plaice.is_continuous(d::WrenDists.IID) = Plaice.is_continuous(WrenDists._dist(d))
Plaice.is_continuous(d::WrenDists.Independent) = all(Plaice.is_continuous, d.components)
Plaice.test_name(d::WrenDists.Independent) = repr(d)
Plaice.rand_safe_ad(d::WrenDists.Independent) = rand(d)
Plaice.can_test_in_support(::WrenDists.IIDUnivariate, x) = true
Plaice.can_test_in_support(::WrenDists.IndependentArray, x) = true
function Plaice.test_in_support(d::WrenDists.IIDUnivariate, x)
    for xi in x
        Plaice.test_in_support(WrenDists._dist(d), xi)
    end
end
function Plaice.test_in_support(d::WrenDists.IndependentArray, x)
    for (di, xi) in zip(d.components, x)
        Plaice.test_in_support(di, xi)
    end
end

const TESTED_ADTYPES = (
    "ForwardDiff" => DI.AutoForwardDiff(),
    "ReverseDiff" => DI.AutoReverseDiff(),
    "ReverseDiffCpl" => DI.AutoReverseDiff(; compile=true),
    "MooncakeFwd" => DI.AutoMooncakeForward(),
    "MooncakeRvs" => DI.AutoMooncake(),
    "EnzymeFwd" => DI.AutoEnzyme(; mode=EnzymeCore.Forward),
    "EnzymeRvs" => DI.AutoEnzyme(; mode=EnzymeCore.Reverse),
)
const REF_ADTYPE = DI.AutoForwardDiff()

function logp_from_uncons(y, dist)
    finv = Plaice.from_unconstrained_vec(dist)
    x = finv(y)
    # Just checking that the logdensity is differentiable, so no need to include the
    # Jacobian
    return logdensityof(dist, x)
end

function test_ad(t::ADTestCase)
    ref_prep = DI.prepare_gradient(t.f, REF_ADTYPE, t.x, t.contexts...)
    g_ref = DI.gradient(t.f, ref_prep, REF_ADTYPE, t.x, t.contexts...)
    @testset "$adtype_name" for (adtype_name, adtype) in TESTED_ADTYPES
        adtype == REF_ADTYPE || adtype_name in t.skipped_adtypes && continue
        prep = DI.prepare_gradient(t.f, adtype, t.x, t.contexts...)
        g = DI.gradient(t.f, prep, adtype, t.x, t.contexts...)
        @test isapprox(g, g_ref; rtol=1e-6, atol=1e-6)
    end
end

function WrenDists.test_dist(
    name,
    d_wd,
    d_ref;
    plaice_zero_allocs=(),
    test_reactant=true,
    skipped_adtypes=(),
    other_ad_testcases=(),
)
    @testset "$name" begin
        @info "Testing $name"

        rng = Xoshiro(468)

        @testset "logdensityof consistent with Distributions" begin
            for _ in 1:100
                x = rand(rng, d_ref)
                @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
            end
        end

        @testset "logdensityof is differentiable wrt sample" begin
            if !(
                d_wd isa WrenDists.DiscreteUnivariateDist ||
                d_wd isa WrenDists.DiscreteMultivariateDist
            )
                f = logp_from_uncons
                x = rand(rng, Plaice.unconstrained_vec_length(d_wd))
                test_ad(ADTestCase(f, x, (DI.Constant(d_wd),), skipped_adtypes))
            end
        end

        @testset "logdensityof is differentiable wrt params" begin
            for t in other_ad_testcases
                test_ad(t)
            end
        end

        @testset "rand obeys rng" begin
            @test rand(Xoshiro(468), d_wd) == rand(Xoshiro(468), d_wd)
            @test rand(Xoshiro(468), d_wd) != rand(Xoshiro(469), d_wd)
        end

        @testset "type stability" begin
            x = @inferred rand(rng, d_wd)
            @inferred logdensityof(d_wd, x)
        end

        @testset "Plaice" begin
            Plaice.test_all(d_wd; expected_zero_allocs=plaice_zero_allocs)
        end

        @testset "Reactant" begin
            if !test_reactant
                # Marker so that disabled Reactant tests show up as broken rather than
                # silently passing.
                @test_broken false
            else
                points = [rand(rng, d_ref) for _ in 1:10]
                WrenDists.test_dist_reactant(d_wd, points)
            end
        end
    end
end

end
