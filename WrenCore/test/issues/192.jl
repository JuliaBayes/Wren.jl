module Issue192

using Test, WrenCore
using WrenDists: WrenDists as WD
import Enzyme

@wren function positive_flat_scale(y)
    sigma ~ WD.truncated(WD.Flat(); lower=0.0)
    @observe y ~ WD.Normal(0.0, sigma)
end

@testset "#192 positive flat scale" begin
    vm = VecModel(positive_flat_scale(1.0), AutoEnzyme(; mode=Reverse);
        latent_source=FromValues(VarNamedTuple(sigma=2.0)), reduce_copies=false)
    u = log(2.0)
    value, gradient = lp_and_grad(vm, [u])
    @test value ≈ logdensityof(WD.Normal(0.0, 2.0), 1.0) + u
    @test gradient ≈ [0.25]

    prior_vm = VecModel(positive_flat_scale(1.0), AutoEnzyme(; mode=Reverse))
    @test isfinite(lp(prior_vm, vector_parameters(prior_vm)))
end

end
