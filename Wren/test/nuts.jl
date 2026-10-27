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

@testset "nuts.jl" begin
    @testset "interface" begin
        chain = nuts(model, 1000, 4; progress=false)
        @test chain isa VNChain
        @test FlexiChains.niters(chain) == 1000
        @test FlexiChains.nchains(chain) == 4
        @test only(FlexiChains.parameters(chain)) == @vn(x)
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
