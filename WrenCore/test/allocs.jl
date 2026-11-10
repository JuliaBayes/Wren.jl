"""
Check that for simple models, Wren is allocation-free.
"""
module WrenAllocsTests

@info "testing allocs.jl"

using WrenCore
using Distributions
using Test
using Chairmarks

function test_zero_allocs(model)
    @test iszero((@b parameters($model)).allocs)
    v = VecModel(model, nothing)
    x = vector_parameters(v)
    @test iszero((@b lp($v, $x)).allocs)
end

@testset "allocs.jl" begin
    @testset "scalar params only" begin
        @wren function f()
            x ~ Normal()
            y ~ Normal(x)
        end
        test_zero_allocs(f())
    end

    @testset "vector params passed as argument" begin
        @wren function f2(z)
            x ~ Normal()
            y ~ Normal(x)
            for i in 1:5
                z[i] ~ Normal(x)
            end
        end
        model = f2(ones(5)) | (; z=zeros(5))
        test_zero_allocs(model)
    end

    @testset "submodel doesn't introduce allocations" begin
        @wren function inner()
            x ~ Normal()
        end
        @wren function outer(sub)
            a ~ sub
        end
        model = outer(inner())
        test_zero_allocs(model)
    end
end

end # module WrenAllocsTests
