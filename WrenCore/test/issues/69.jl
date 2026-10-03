module Issue69

using WrenCore
using WrenDists
using Enzyme
using FiniteDifferences
using Random
using Test
using DifferentiationInterface

@testset "#69 Enzyme.Const differentiable data" begin
    @wren function f(x, y, N)
        @batch for i in 1:N
            x[i] ~ Normal()
            @observe y[i] ~ Normal(x[i])
        end
    end

    N = 10
    x, y = randn(N), randn(N)
    model = f(x, y, N)

    vm1 = VecModel(model, WrenCore.DEFAULT_AD)
    vm2 = VecModel(model, AutoFiniteDifferences(; fdm=central_fdm(5, 1)))

    for i in 1:5
        params = rand(vm1)
        lp1, grad1 = lp_and_grad(vm1, params)
        lp2, grad2 = lp_and_grad(vm2, params)

        @test lp1 ≈ lp2
        @test grad1 ≈ grad2
    end
end

end
