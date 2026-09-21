module WrenBatchTypeStabilityTests

@info "testing batch/typestability.jl"

using Test
using WrenCore
using WrenDists

@testset "batch type stability: TDist" begin
    iters = 1:3
    μ_idx = WrenCore.Indexed{1,1}([1.0, 2.0, 3.0])
    σ_idx = WrenCore.Indexed{1,1}([0.5, 1.0, 2.0])
    ν_idx = WrenCore.Indexed{1,1}([3.0, 4.0, 5.0])

    # 3-arg: all combinations of Number / Indexed
    @testset "TDist(Number, Indexed, Indexed)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), 4.0, μ_idx, σ_idx)
    end
    @testset "TDist(Number, Indexed, Number)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), 4.0, μ_idx, 2.0)
    end
    @testset "TDist(Number, Number, Indexed)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), 4.0, 0.0, σ_idx)
    end
    @testset "TDist(Indexed, Indexed, Indexed)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), ν_idx, μ_idx, σ_idx)
    end
    @testset "TDist(Indexed, Indexed, Number)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), ν_idx, μ_idx, 2.0)
    end
    @testset "TDist(Indexed, Number, Indexed)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), ν_idx, 0.0, σ_idx)
    end
    @testset "TDist(Indexed, Number, Number)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), ν_idx, 0.0, 1.0)
    end

    # 2-arg
    @testset "TDist(Number, Indexed)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), 4.0, μ_idx)
    end
    @testset "TDist(Indexed, Indexed) 2-arg" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), ν_idx, μ_idx)
    end

    # 1-arg
    @testset "TDist(Indexed)" begin
        @inferred WrenCore.batched(iters, Val{TDist}(), ν_idx)
    end
end

end # module WrenBatchTypeStabilityTests
