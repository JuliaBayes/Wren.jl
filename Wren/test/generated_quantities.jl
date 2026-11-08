module WrenGeneratedQuantitiesTests

using Wren
using Wren.Dists
using FlexiChains
using Test
import Enzyme

@wren function gq_model()
    x ~ Normal()
    @gq x2 = x^2
end
model = gq_model()

@testset "generated quantities in sampler chains" begin
    @testset "prior" begin
        chain = prior(model, 10; progress=false)
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x), @vn(x2)])
        @test chain[@vn(x2)] ≈ chain[@vn(x)] .^ 2

        chain = prior(model, 10; progress=false, gqs=None())
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x)])
    end

    @testset "nuts" begin
        chain = nuts(model, 20; progress=false)
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x), @vn(x2)])
        @test chain[@vn(x2)] ≈ chain[@vn(x)] .^ 2

        chain = nuts(model, 20; progress=false, gqs=Only(()))
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x)])
    end
end

end
