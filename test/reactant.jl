module WrenReactantTests

using Wren
using ProbabilityMeasures: ProbabilityMeasures as PM
using Enzyme: Reverse
using Reactant
using ADTypes: AutoReactant, AutoEnzyme
using Test

@testset "reactant.jl" begin
    @testset "basic" begin
        # equivalent Wren model
        @wren function f()
            x ~ PM.Normal(0.0, 1.0)
        end
        model = f()
        vm = VecModel(model, AutoReactant(AutoEnzyme(; mode=Reverse)))
        params = [0.0]
        @test lp(vm, params) ≈ logdensityof(PM.Normal(0.0, 1.0), 0.0)
    end
end

end # module
