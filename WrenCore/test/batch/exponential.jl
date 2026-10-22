module WrenBatchExponentialTests

@info "testing batch/exponential.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.Exponential" begin
    @testset "batched: Exponential(theta[i]) -> BatchedExponential" begin
        @wren function theta_batch(x, theta)
            @batch for i in 1:3
                x[i] ~ WrenDists.Exponential(theta[i])
            end
        end
        @wren function theta_nobatch(x, theta)
            for i in 1:3
                x[i] ~ WrenDists.Exponential(theta[i])
            end
        end
        theta = [0.5, 1.0, 2.0]
        test_model_equivalence(theta_batch(ones(3), theta), theta_nobatch(ones(3), theta))
    end
end

end # module WrenBatchExponentialTests
