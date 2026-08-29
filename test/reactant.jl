module WrenReactantTests

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

        for dist in (PM.Normal(0.0, 1.0),)
            model = f(dist)
            vm = VecModel(model, AutoReactant(AutoEnzyme(; mode=Enzyme.Reverse)))
            params = [0.0]
            true_lp, true_grad = true_logdens_and_grad(dist, params)
            @test lp(vm, params) ≈ true_lp
            l, grad = lp_and_grad(vm, params)
            @test l ≈ true_lp
            @test grad ≈ true_grad
        end
    end
end

end # module
