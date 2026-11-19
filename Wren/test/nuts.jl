module WrenNUTSTests

using Wren
using Wren.Dists
using FlexiChains
using Test
using Random
import Enzyme
import ForwardDiff

@wren function f()
    x ~ Normal()
end
model = f()

struct RecordStarts
    starts::Channel{Vector{Float64}}
end
function Wren.DynamicHMC.warmup(_, recorder::RecordStarts, state)
    put!(recorder.starts, copy(state.Q.q))
    return nothing, state
end

@testset "nuts.jl" begin
    @testset "interface" begin
        chain = nuts(model, 1000, 4; progress=false)
        @test chain isa VNChain
        @test FlexiChains.niters(chain) == 1000
        @test FlexiChains.nchains(chain) == 4
        @test only(FlexiChains.parameters(chain)) == @vn(x)
    end

    @testset "initialization" begin
        @wren function bounded_posterior()
            x ~ Normal(10.0, 0.1)
            @addloglikelihood (9.0 < x < 11.0 ? 0.0 : -Inf)
        end
        starts = Channel{Vector{Float64}}(2)
        kwargs = (;
            progress=false,
            warmup_stages=(RecordStarts(starts), Wren.DynamicHMC.InitialStepsizeSearch()),
        )
        chn1 = nuts(Xoshiro(468), bounded_posterior(), 3, 2; kwargs...)
        q1, q2 = take!(starts), take!(starts)
        chn2 = nuts(Xoshiro(468), bounded_posterior(), 3, 2; kwargs...)
        @test all(x -> 9.0 < x < 11.0, chn1[@vn(x)])
        @test FlexiChains.has_same_data(chn1, chn2)
        @test q1 != q2

        chn = nuts(
            Xoshiro(468),
            model,
            3;
            progress=false,
            initialization=(; q=[10.0], ϵ=0.01),
            warmup_stages=(),
            algorithm=Wren.DynamicHMC.NUTS(; max_depth=1),
        )
        @test all(x -> 9.8 < x < 10.2, chn[@vn(x)])
    end

    @testset "Float32 prior" begin
        @wren f32() = x ~ Normal(0.0f0, 1.0f0)
        chain = nuts(Xoshiro(4), f32(), 50; adtype=AutoForwardDiff(), progress=false)
        @test FlexiChains.niters(chain) == 50
        @test all(isfinite, chain[@vn(x)])
    end

    @testset "rng" begin
        chn1 = nuts(Xoshiro(468), model, 100, 3; progress=false)
        chn2 = nuts(Xoshiro(468), model, 100, 3; progress=false)
        @test FlexiChains.has_same_data(chn1, chn2)
        chn3 = nuts(Xoshiro(469), model, 100, 3; progress=false)
        @test !FlexiChains.has_same_data(chn1, chn3)

        # reusing RNG should give new chains
        s = Xoshiro(468)
        chn1 = nuts(s, model, 100, 3; progress=false)
        chn2 = nuts(s, model, 100, 3; progress=false)
        @test !FlexiChains.has_same_data(chn1, chn2)
    end
end

end
