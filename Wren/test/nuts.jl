module WrenNUTSTests

using Wren
using Wren.Dists
using FlexiChains
using Test
using Random
import Enzyme

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

struct RecordInitialization
    states::Channel{NamedTuple}
end
function Wren.DynamicHMC.warmup(_, recorder::RecordInitialization, state)
    put!(recorder.states, (; q=copy(state.Q.q), state.κ, state.ϵ))
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

    @testset "model-space initialization" begin
        @wren function positive_parameter()
            x ~ LogNormal()
        end
        states = Channel{NamedTuple}(2)
        κ = Wren.DynamicHMC.GaussianKineticEnergy(1, 2.0)
        kwargs = (;
            progress=false,
            initialization=(; κ, ϵ=0.01),
            warmup_stages=(RecordInitialization(states),),
            algorithm=Wren.DynamicHMC.NUTS(; max_depth=1),
        )
        initial = VarNamedTuple(; x=exp(2.0))
        nuts(Xoshiro(468), positive_parameter(), 3, 2; init=initial, kwargs...)
        s1, s2 = take!(states), take!(states)
        @test s1.q ≈ s2.q ≈ [2.0]
        @test s1.κ == s2.κ == κ
        @test s1.ϵ == s2.ϵ == 0.01
        chn = nuts(Xoshiro(468), positive_parameter(), 3; init=initial, progress=false)
        @test all(>(0), chn[@vn(x)])

        nuts(
            Xoshiro(468),
            positive_parameter(),
            3,
            2;
            init=FromPrior(Xoshiro(123)),
            kwargs...,
        )
        q1, q2 = take!(states).q, take!(states).q
        nuts(
            Xoshiro(468),
            positive_parameter(),
            3,
            2;
            init=FromPrior(Xoshiro(123)),
            kwargs...,
        )
        @test q1 != q2
        @test sort([q1, q2]) == sort([take!(states).q, take!(states).q])

        @test_throws ArgumentError nuts(
            positive_parameter(),
            3;
            init=initial,
            initialization=(; q=[0.0]),
            progress=false,
        )
    end
end

end
