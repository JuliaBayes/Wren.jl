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

    vm_fd = VecModel(model, AutoFiniteDifferences(; fdm=central_fdm(5, 1)))

    # Make sure that `reduce_copies` doesn't jeopardise correctness.
    for reduce_copies in (true, false)
        vm_enz = VecModel(model, WrenCore.DEFAULT_AD; reduce_copies=reduce_copies)

        for i in 1:5
            params = rand(vm_enz)
            lp_enz, grad_enz = lp_and_grad(vm_enz, params)
            lp_fd, grad_fd = lp_and_grad(vm_fd, params)

            @test lp_enz ≈ lp_fd
            @test grad_enz ≈ grad_fd
        end
    end

    @testset "reduce_copies" begin
        # Dataflow analysis should detect that `y` doesn't need to be copied
        vm_full = VecModel(model, WrenCore.DEFAULT_AD; reduce_copies=false)
        @test WrenCore._copied_args(vm_full.model) == (:N, :x)

        # reduce_copies shouldn't reduce that further
        vm_reduced = VecModel(model, WrenCore.DEFAULT_AD; reduce_copies=true)
        @test WrenCore._copied_args(vm_reduced.model) == (:N, :x)
    end

    @testset "with nocopy" begin
        # This should give incorrect gradients with Enzyme.
        # https://github.com/EnzymeAD/Enzyme.jl/issues/2344
        #
        # Technically this is not a Wren test, and in principle if Enzyme magically fixes
        # the const 'issue' this test will fail. But given that it's documented /
        # intentional behaviour in Enzyme, I think this should be reasonably reliable to
        # keep.
        @wren function f_nocopy(@nocopy(x), y, N)
            @batch for i in 1:N
                x[i] ~ Normal()
                @observe y[i] ~ Normal(x[i])
            end
        end
        N = 10
        x, y = randn(N), randn(N)
        model_nocopy = f_nocopy(x, y, N)

        vm_nocopy = VecModel(model_nocopy, WrenCore.DEFAULT_AD)
        params = rand(vm_nocopy)

        _, grad_nocopy = lp_and_grad(vm_nocopy, params)
        _, grad_fd = lp_and_grad(vm_fd, params)

        # Should be wrong.
        @test maximum(abs.(grad_nocopy .- grad_fd)) > 1e-4
    end

end

end
