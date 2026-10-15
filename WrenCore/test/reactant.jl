module WrenReactantTests

@info "testing reactant.jl"

using ADTypes: AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using Test
using WrenCore
using WrenDists

@wren function reactant_model(a)
    x ~ Normal(a, 1.0)
end

@testset "Reactant" begin
    adtype = AutoReactant(AutoEnzyme(; mode=Reverse))
    vecmodel = VecModel(reactant_model(1.0), adtype; reduce_copies=true)
    params = rand(vecmodel)

    @test lp(vecmodel, params) isa Float64
    logdensity, grad = lp_and_grad(vecmodel, params)
    @test logdensity isa Float64
    @test grad isa Vector{Float64}
end

end # module WrenReactantTests
