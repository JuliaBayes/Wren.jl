module WrenReactantTests

@info "testing reactant.jl"

using Wren
using Plaice: Plaice
using ProbabilityMeasures: ProbabilityMeasures as PM
using Enzyme: Enzyme
using ADTypes: AutoReactant, AutoEnzyme
using Test

import Pkg
Pkg.add("Reactant")
using Reactant: Reactant

@testset "reactant.jl" begin
    @testset "PM distributions" begin
        @wren function f(d)
            x ~ d
        end

        function true_logdens(d, yvec)
            tfm = Plaice.from_unconstrained_vec(d)
            x, inv_logjac = Plaice.with_logabsdet_jacobian(tfm, yvec)
            return logdensityof(d, x) + inv_logjac
        end
        function true_logdens_and_grad(d, yvec)
            r = Enzyme.gradient(
                Enzyme.ReverseWithPrimal,
                true_logdens,
                Enzyme.Const(d),
                yvec,
            )
            return (r.val, r.derivs[2])
        end

        @testset "$(nameof(typeof(dist)))" for dist in (
            # The idea in curating this list of distributions is that we probably want to
            # test each distribution that has a different bijector. We don't need to test
            # the log-density of every distribution since PM already tests Reactant
            # compatibility with that. In principle, Plaice also tests Reactant
            # compatibility on the bijectors, but it's probably worth doing that testing
            # here as well since we are doing a bit more than just transforming the sample.
            PM.Normal(), # identity
            PM.LogNormal(), # positive
            PM.Uniform(0.0, 1.0), # truncated
            PM.MvNormal([0.0, 0.0], I), # identity multivariate
            Wren.iid(PM.Normal(0.5, 2.0), (3,)),
            Wren.iid(PM.Exponential(1.5), (2, 2)),
            Wren.iid(PM.Uniform(-1.0, 2.0), (4,)),
            Wren.iid(PM.MvNormal([0.0, 0.0], I), (2,)),
        )
            model = f(dist)
            vm = VecModel(model, AutoReactant(AutoEnzyme(; mode=Enzyme.Reverse)))
            params = rand(vm)
            true_lp, true_grad = true_logdens_and_grad(dist, params)
            @test lp(vm, params) ≈ true_lp
            l, grad = lp_and_grad(vm, params)
            @test l ≈ true_lp
            @test grad ≈ true_grad
        end
    end
end

end # module
