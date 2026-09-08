module WrenAliasingTests

@info "testing aliasing.jl"

using WrenCore
using Distributions
using Test

@testset "aliasing.jl" begin
    @testset "type stability of _dealias" begin
        @inferred WrenCore._dealias(1)
        @inferred WrenCore._dealias(:hello)
        @inferred WrenCore._dealias("hello")
        @inferred WrenCore._dealias(randn(10))
        @inferred WrenCore._dealias(view(randn(10), 1:5))
        @inferred WrenCore._dealias(view(fill("hi", 10), 1:5))
    end

    @testset "model construction copies args" begin
        @wren function fassume(x)
            a ~ Normal()
            for i in eachindex(x)
                x[i] ~ Normal(a)
            end
        end
        @wren function fobserve(x)
            a ~ Normal()
            for i in eachindex(x)
                @observe x[i] ~ Normal(a)
            end
        end

        x = randn(10)
        xcopy = deepcopy(x)

        @testset "assumed x" begin
            model = fassume(x)
            @test model.args.x !== x # Should be dealiased.
            rand(model) # Run the model
            # Contents should not have been modified, because the model writes
            # into a copy of x.
            @test x == xcopy
        end

        @testset "observed x" begin
            model = fobserve(x)
            @test model.args.x !== x
            rand(model)
            @test x == xcopy
        end

        @testset "conditioned x" begin
            model = fassume(x) | VarNamedTuple(x=x)
            @test model.args.x !== x # Should be dealiased.
            @test model.conditioned[@vn(x)] === x # Should be the same.
            rand(model) # Run the model
            # Contents should not have been modified, because conditioned values are
            # read-only.
            @test x == xcopy
        end
    end
end

end
