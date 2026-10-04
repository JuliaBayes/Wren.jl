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

    @testset "@nocopy" begin
        @testset "removes argument from the copy set" begin
            # `y` is written to and then read, so it would normally be copied.
            @wren function nc_default(y)
                y[1] ~ Normal()
                return y
            end
            @wren function nc_annotated(@nocopy(y))
                y[1] ~ Normal()
                return y
            end
            @test WrenCore._copied_args(nc_default([0.0])) == (:y,)
            @test WrenCore._copied_args(nc_annotated([0.0])) == ()
        end

        @testset "other arguments are still analysed" begin
            @wren function nc_mixed(x, @nocopy(y), z)
                x[1] ~ Normal()
                y[1] ~ Normal()
                @observe z ~ Normal(x[1] + y[1])
            end
            @test WrenCore._copied_args(nc_mixed([0.0], [0.0], 0.0)) == (:x,)
        end

        @testset "qualified macro name and where clauses" begin
            @wren function nc_qualified(WrenCore.@nocopy(y))
                y[1] ~ Normal()
                return y
            end
            @test WrenCore._copied_args(nc_qualified([0.0])) == ()

            sig, nocopy_args =
                WrenCore.strip_nocopy_annotations(:(f(x, @nocopy(y)) where {T}))
            @test sig == :(f(x, y) where {T})
            @test nocopy_args == Set([:y])
        end

        @testset "evaluation writes into the uncopied argument" begin
            @wren function nc_write(@nocopy(y))
                y[1] ~ Normal()
                return y
            end
            @wren function copy_write(y)
                y[1] ~ Normal()
                return y
            end

            model = copy_write([0.0])
            rand(model)
            @test model.args.y == [0.0]

            model = nc_write([0.0])
            rand(model)
            @test model.args.y != [0.0]
        end

        @testset "copy set is kept when conditioning" begin
            @wren function nc_cond(x, @nocopy(y))
                a ~ Normal()
                x[1] ~ Normal(a)
                y[1] ~ Normal(a)
                return (x, y)
            end
            model = nc_cond([0.0], [0.0])
            @test WrenCore._copied_args(model) == (:x,)
            conditioned_model = model | VarNamedTuple(a=1.0)
            @test WrenCore._copied_args(conditioned_model) == (:x,)
            @test WrenCore._copied_args(decondition(conditioned_model)) == (:x,)
        end

        @testset "errors" begin
            # Invalid uses inside the signature
            for sig in (:(f(@nocopy(y::Vector))), :(f(@nocopy(x, y))), :(f(@nocopy())))
                @test_throws "@nocopy` must be applied to a single argument name" WrenCore._wren(
                    Expr(:function, sig, :(x ~ Normal())),
                    @__MODULE__,
                )
            end
            # Outside of a @wren signature
            @test_throws "@nocopy must be used" @macroexpand @nocopy(y)
        end
    end

    @testset "copy set is a type parameter of the evaluation function" begin
        @wren function shared_rgf(y)
            y[1] ~ Normal()
            return y
        end
        model = shared_rgf([0.0])
        @test WrenCore._copied_args(model) == (:y,)

        # Build a model which reuses the same RGF but copies nothing.
        rgf = model.evaluation_fn.rgf
        nocopy_model = WrenCore.Model(
            model.name,
            model.args,
            model.stmts,
            model.mod;
            _unsafe_reuse_evaluation_fn=WrenCore.EvaluationFn{()}(rgf),
        )
        @test nocopy_model.evaluation_fn.rgf === rgf
        @test WrenCore._copied_args(nocopy_model) == ()

        # The copy set controls whether evaluation writes into `model.args`.
        rand(model)
        @test model.args.y == [0.0]
        rand(nocopy_model)
        @test nocopy_model.args.y != [0.0]

        # Equal sets give equal types, regardless of the order the set was built in.
        @test typeof(WrenCore.EvaluationFn(rgf, Set([:a, :b]))) ===
              typeof(WrenCore.EvaluationFn(rgf, Set([:b, :a])))
    end
end

end
