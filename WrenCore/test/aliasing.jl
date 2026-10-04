module WrenAliasingTests

@info "testing aliasing.jl"

using WrenCore
using Distributions
using Test

# The names of the arguments that `model` copies on every evaluation.
copied_args(::WrenCore.Model{A,C,F,Val{S}}) where {A,C,F,S} = S

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
        @testset "strip_nocopy_annotations" begin
            strip_nc = WrenCore.strip_nocopy_annotations
            function test_strip(sig, expected_sig, expected_nocopy)
                original = deepcopy(sig)
                new_sig, nocopy_args = strip_nc(sig)
                @test new_sig == expected_sig
                @test nocopy_args == Set{Symbol}(expected_nocopy)
                @test sig == original # input should not be mutated
            end

            @testset "no annotations" begin
                test_strip(:(f()), :(f()), [])
                test_strip(:(f(x)), :(f(x)), [])
                test_strip(:(f(x, y)), :(f(x, y)), [])
            end

            @testset "annotated arguments" begin
                test_strip(:(f(@nocopy(y))), :(f(y)), [:y])
                test_strip(:(f(x, @nocopy(y))), :(f(x, y)), [:y])
                test_strip(:(f(@nocopy(x), y)), :(f(x, y)), [:x])
                test_strip(:(f(@nocopy(x), @nocopy(y))), :(f(x, y)), [:x, :y])
                # Argument order is preserved.
                test_strip(:(f(@nocopy(x), y, @nocopy(z))), :(f(x, y, z)), [:x, :z])
            end

            @testset "macro call without parentheses" begin
                test_strip(:(f(@nocopy y)), :(f(y)), [:y])
                test_strip(:(f(x, @nocopy y)), :(f(x, y)), [:y])
            end

            @testset "where clauses" begin
                test_strip(:(f(x, @nocopy(y)) where {T}), :(f(x, y) where {T}), [:y])
                test_strip(
                    :(f(@nocopy(y)) where {T} where {U}),
                    :(f(y) where {T} where {U}),
                    [:y],
                )
            end

            @testset "other arguments are left alone" begin
                # Other macros, including qualified forms of `@nocopy`, are not touched.
                test_strip(:(f(@other(y))), :(f(@other(y))), [])
                test_strip(:(f(WrenCore.@nocopy(y))), :(f(WrenCore.@nocopy(y))), [])
                # Non-symbol arguments are passed through (`extract_args` rejects them
                # later).
                test_strip(:(f(x::Int, @nocopy(y))), :(f(x::Int, y)), [:y])
            end

            @testset "errors" begin
                msg = "@nocopy` must be applied to a single argument name"
                @test_throws msg strip_nc(:(f(@nocopy(y::Vector))))
                @test_throws msg strip_nc(:(f(@nocopy())))
                @test_throws msg strip_nc(:(f(@nocopy(x, y))))
                @test_throws msg strip_nc(:(f(@nocopy(@nocopy(y)))))
                # Without parentheses, the macro takes everything after it, so this is
                # `@nocopy((y, z))`.
                @test_throws msg strip_nc(:(f(@nocopy y, z)))
                # Not a function signature
                @test_throws "expected function signature" strip_nc(:(x = 1))
            end
        end

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
            @test copied_args(nc_default([0.0])) == (:y,)
            @test copied_args(nc_annotated([0.0])) == ()
        end

        @testset "other arguments are still analysed" begin
            @wren function nc_mixed(x, @nocopy(y), z)
                x[1] ~ Normal()
                y[1] ~ Normal()
                @observe z ~ Normal(x[1] + y[1])
            end
            @test copied_args(nc_mixed([0.0], [0.0], 0.0)) == (:x,)
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
            # If `copied_args` doesn't contain `:y`, then the subsequent test doesn't make
            # sense.
            @test copied_args(model) == (:y,)
            rand(model)
            @test model.args.y == [0.0]

            model = nc_write([0.0])
            @test copied_args(model) == () # likewise.
            rand(model)
            @test model.args.y != [0.0]
        end

        @testset "copy set is kept when conditioning" begin
            # TODO(penelopeysm): This might change if we make conditioning smarter.
            # https://github.com/JuliaBayes/Wren.jl/issues/139
            @wren function nc_cond(x, @nocopy(y))
                a ~ Normal()
                x[1] ~ Normal(a)
                y[1] ~ Normal(a)
                return (x, y)
            end
            model = nc_cond([0.0], [0.0])
            @test copied_args(model) == (:x,)
            conditioned_model = model | VarNamedTuple(a=1.0)
            @test copied_args(conditioned_model) == (:x,)
            @test copied_args(decondition(conditioned_model)) == (:x,)
        end
    end

    @testset "changing copy set directly" begin
        @wren function shared_rgf(y)
            y[1] ~ Normal()
            return y
        end
        model = shared_rgf([0.0])
        @test copied_args(model) == (:y,)

        # Build a model which reuses the same evaluation function but copies nothing.
        nocopy_model = WrenCore.Model(
            model.name,
            model.args,
            model.stmts,
            model.mod;
            copy_argnames=Val(()),
            _unsafe_reuse_evaluation_fn=model.evaluation_fn,
        )
        @test nocopy_model.evaluation_fn === model.evaluation_fn
        @test copied_args(nocopy_model) == ()

        # The copy set controls whether evaluation writes into `model.args`.
        rand(model)
        @test model.args.y == [0.0]
        rand(nocopy_model)
        @test nocopy_model.args.y != [0.0]
    end

    @testset "only arguments that can hold differentiable data are copied" begin
        maybe_dealias(v) = WrenCore._maybe_dealias(Val((:v,)), Val(:v), v)
        # Copied
        v = [1.0, 2.0]
        @test maybe_dealias(v) !== v
        # Not copied
        for v in ([1, 2], [true, false], view([1, 2, 3], 1:2), 1, (1.0, 2.0))
            @test maybe_dealias(v) === v
        end
        # Not in the copy set, so not copied even though it could hold differentiable data
        v = [1.0, 2.0]
        @test WrenCore._maybe_dealias(Val((:w,)), Val(:v), v) === v

        @testset "integer arrays are still copied on model construction" begin
            @wren function int_arg(z)
                a ~ Normal()
                return z
            end
            z = [1, 2]
            @test int_arg(z).args.z !== z
        end
    end
end

end
