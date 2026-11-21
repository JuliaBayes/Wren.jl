module WrenTrackedQuantitiesTests

using Wren
using Wren.Dists
using FlexiChains
using Test
import Enzyme

@wren function tq_model()
    x ~ Normal()
    @track x2 = x^2
end
model = tq_model()

@testset "tracked quantities in sampler chains" begin
    @testset "prior" begin
        chain = prior(model, 10; progress_backend=())
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x), @vn(x2)])
        @test chain[@vn(x2)] ≈ chain[@vn(x)] .^ 2

        chain = prior(model, 10; progress_backend=(), track=NoQuantities())
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x)])
    end

    @testset "nuts" begin
        chain = nuts(model, 20; progress_backend=())
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x), @vn(x2)])
        @test chain[@vn(x2)] ≈ chain[@vn(x)] .^ 2

        chain = nuts(model, 20; progress_backend=(), track=SomeQuantities(()))
        @test Set(FlexiChains.parameters(chain)) == Set([@vn(x)])
    end
end

end
