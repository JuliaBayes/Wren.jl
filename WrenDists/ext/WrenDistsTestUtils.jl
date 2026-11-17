module WrenDistsTestUtils

using WrenDists: WrenDists, logdensityof, ADTestCase
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using Distributions: logpdf
using EnzymeCore: EnzymeCore
using Plaice: Plaice
using Random: Xoshiro
using Test

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
    scenarios = [
        DIT.Scenario{:gradient,place}(
            t.f,
            t.x,
            t.contexts...;
            res1=g_ref,
            # Distribution contexts need not implement zero.
            prep_args=(; x=t.x, contexts=t.contexts),
        ) for place in (:in, :out)
    ]
    backends = [
        adtype for (adtype_name, adtype) in TESTED_ADTYPES if
        adtype == REF_ADTYPE || !(adtype_name in t.skipped_adtypes)
    ]
    DIT.test_differentiation(
        backends,
        scenarios;
        rtol=1e-6,
        atol=1e-6,
        # Array-bearing distributions need not compare equal after deepcopy.
        scenario_intact=false,
    )
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
    @testset "test_dist: $name" begin
        @info "test_dist: $name"

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
