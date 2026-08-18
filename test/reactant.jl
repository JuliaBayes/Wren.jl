module WrenReactantTests

using Wren
using ChangesOfVariables: ChangesOfVariables
using InverseFunctions: InverseFunctions
using ProbabilityMeasures: ProbabilityMeasures as PM
using Enzyme: Reverse
using Reactant
using ADTypes: AutoReactant, AutoEnzyme
using Test

struct Only end
Only(x::AbstractVector{<:Number}) = x[]
ChangesOfVariables.with_logabsdet_jacobian(::Only, x::AbstractVector{<:Number}) = (x[], zero(eltype(x)))
InverseFunctions.inverse(::Only) = Vec()
struct Vec end
Vec(r) = [r]
ChangesOfVariables.with_logabsdet_jacobian(::Vec, v) = ([v], zero(eltype(v)))
InverseFunctions.inverse(::Vec) = Only()
Wren.from_unconstrained_vec(::PM.Normal) = Only()
Wren.from_vec(::PM.Normal) = Only()

@testset "reactant.jl" begin
    @testset "basic" begin
        # equivalent non-Wren function
        function fbasic(x)
            return logdensityof(PM.Normal(0.0, 1.0), x[])
        end
        x = [0.0]
        rx = Reactant.to_rarray(x)
        fbasic2 = @allowscalar @compile sync=true fbasic(rx)
        from_rarray(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)
        function fbasic3(x)
            return from_rarray(fbasic2(Reactant.to_rarray(x)))
        end
        @be fbasic3($x)

        # equivalent Wren model
        @model function f()
            x ~ PM.Normal(0.0, 1.0)
        end
        model = f()
        vm = VecModel(model, AutoReactant(AutoEnzyme(; mode=Reverse)));
        params = [0.0]
        @test lp(vm, params) ≈ f(params)
        @be lp($vm, $params)
        @be lp_and_grad($vm, $params)
    end
end

end # module
