module WrenSyntaxTests

using Wren
using Test
using Distributions
using DensityInterface: logdensityof
using Chairmarks
using VarNames

macro preserve(expr)
    return esc(expr)
end

@testset "syntax.jl" begin
    @testset "basic models" begin
        @testset "1x sample" begin
            @wren function simple_normal()
                x ~ Normal()
                return nothing
            end
            model = simple_normal()
            @test model.name === :simple_normal
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = parameters(accs)
            @test retval === nothing
            @test pars isa VarNamedTuple
            @test only(keys(pars)) == @vn(x)
            @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x)])
            @test iszero(loglikelihood(accs))
        end

        @testset "1x sample with model argument" begin
            @wren function sample_with_arg(m)
                x ~ Normal(m[1])
                return nothing
            end
            d = [100.0]
            model = sample_with_arg(d)
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            @test retval === nothing
            @test parameters(accs) isa VarNamedTuple
            @test logprior(accs) ≈ logdensityof(Normal(d[1]), parameters(accs)[@vn(x)])
            @test iszero(loglikelihood(accs))
        end
    end

    @testset "conditioning" begin
        @wren function x_and_y()
            x ~ Normal()
            y ~ Normal(x)
            return nothing
        end

        @testset "before conditioning" begin
            model = x_and_y()
            @test model.name === :x_and_y
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = parameters(accs)
            @test retval === nothing
            @test pars isa VarNamedTuple
            @test collect(keys(pars)) == [@vn(x), @vn(y)]
            @test logprior(accs) ≈
                  logdensityof(Normal(), pars[@vn(x)]) +
                  logdensityof(Normal(pars[@vn(x)]), pars[@vn(y)])
            @test iszero(loglikelihood(accs))
        end

        @testset "after conditioning" begin
            conditioned_model = x_and_y() | VarNamedTuple(y=1.0)
            @test conditioned_model.name === :x_and_y # shouldn't be changed.
            retval, accs = evaluate(
                conditioned_model,
                FromPrior(),
                OriginalSpace(),
                Wren.default_accumulators(),
            )
            pars = parameters(accs)
            @test retval === nothing
            @test pars isa VarNamedTuple
            @test collect(keys(pars)) == [@vn(x)]
            @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x)])
            @test loglikelihood(accs) ≈ logdensityof(Normal(pars[@vn(x)]), 1.0)
        end
    end

    @testset "observe macro" begin
        @wren function observe_macro()
            x ~ Normal()
            y = x + 1
            @observe y ~ Normal()
            return nothing
        end
        model = observe_macro()
        @test model.name === :observe_macro
        retval, accs =
            evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
        pars = parameters(accs)
        @test retval === nothing
        @test pars isa VarNamedTuple
        @test collect(keys(pars)) == [@vn(x)]
        @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x)])
        @test loglikelihood(accs) ≈ logdensityof(Normal(), pars[@vn(x)] + 1)
    end

    @testset "macros inside models" begin
        @wren function with_macros()
            @preserve begin
                x ~ Normal()
                y = 0.25
                @observe y ~ Normal()
                xs = Vector{Float64}(undef, 2)
                @inbounds for i in eachindex(xs)
                    xs[i] ~ Normal()
                end
            end
        end
        params = VarNamedTuple(; x=0.5, xs=[1.0, 2.0])
        source = FromParams(params)
        _, accs =
            evaluate(with_macros(), source, OriginalSpace(), Wren.default_accumulators())
        @test logprior(accs) ≈ sum(logdensityof.(Ref(Normal()), [0.5, 1.0, 2.0]))
        @test loglikelihood(accs) ≈ logdensityof(Normal(), 0.25)
    end

    @testset "nested expressions" begin
        @wren function nested_if(flag)
            x = if flag
                y ~ Normal()
                y
            else
                z ~ Normal()
                z
            end
            @observe x ~ Normal()
        end

        params = VarNamedTuple(; y=0.5, z=2.0)
        source = FromParams(params)

        _, accs =
            evaluate(nested_if(true), source, OriginalSpace(), Wren.default_accumulators())
        @test logprior(accs) ≈ logdensityof(Normal(), 0.5)
        @test loglikelihood(accs) ≈ logdensityof(Normal(), 0.5)

        _, accs =
            evaluate(nested_if(false), source, OriginalSpace(), Wren.default_accumulators())
        @test logprior(accs) ≈ logdensityof(Normal(), 2.0)
        @test loglikelihood(accs) ≈ logdensityof(Normal(), 2.0)
    end

    @testset "immutable struct" begin
        struct Foo
            a::Float64
        end
        @wren function immutable_struct()
            x = Foo(0.5)
            x.a ~ Normal()
        end
        _, accs = evaluate(
            immutable_struct(),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        pars = parameters(accs)
        @test only(keys(pars)) == @vn(x.a)
        @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x.a)])
    end

    @testset "a = b ~ c syntax" begin
        @testset "macro parses it correctly" begin
            # Essentially we want `a = b ~ c` to be parsed as `a = Sample(b, c)`.
            @wren function parser_inner()
                return 1
            end

            @wren function tilde_value_position()
                retval = x ~ parser_inner()
            end

            stmt =
                only(filter(x -> x isa Wren.AbstractWrenExpr, tilde_value_position().stmts))
            @test stmt isa Wren.WrenExpr
            @test stmt.head == :(=)
            @test stmt.args[1] == :retval
            @test stmt.args[2] isa Wren.Sample
            @test stmt.args[2].lhs == :x
            @test stmt.args[2].rhs isa Wren.WrenExpr
            @test stmt.args[2].rhs.head == :call
            @test stmt.args[2].rhs.args == [:parser_inner]
        end

        @testset "with distributions" begin
            @wren function distribution_tilde_value()
                retval = x ~ Normal()
                @observe retval ~ Normal()
            end

            x = 0.5
            source = FromParams(VarNamedTuple(; x))
            _, accs = evaluate(
                distribution_tilde_value(),
                source,
                OriginalSpace(),
                Wren.default_accumulators(),
            )
            @test parameters(accs) == VarNamedTuple(; x)
            @test logprior(accs) ≈ logdensityof(Normal(), x)
            @test loglikelihood(accs) ≈ logdensityof(Normal(), x)

            conditioned_model = distribution_tilde_value() | VarNamedTuple(; x)
            _, accs = evaluate(
                conditioned_model,
                FromPrior(),
                OriginalSpace(),
                Wren.default_accumulators(),
            )
            @test parameters(accs) == VarNamedTuple()
            @test iszero(logprior(accs))
            @test loglikelihood(accs) ≈ 2 * logdensityof(Normal(), x)
        end
    end


    @testset "@addloglikelihood" begin
        @testset "anonymous" begin
            @wren function addloglik_anon()
                x ~ Normal()
                @addloglikelihood -0.5 * x^2
            end
            model = addloglik_anon()
            _, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = parameters(accs)
            x_val = pars[@vn(x)]
            @test logprior(accs) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(accs) ≈ -0.5 * x_val^2
        end

        @testset "named" begin
            @wren function addloglik_named()
                x ~ Normal()
                @addloglikelihood name=:penalty -x^2
            end
            model = addloglik_named()
            _, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = parameters(accs)
            x_val = pars[@vn(x)]
            @test logprior(accs) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(accs) ≈ -x_val^2
        end

        @testset "multiple addloglikelihood statements" begin
            @wren function addloglik_multiple()
                x ~ Normal()
                @addloglikelihood -x^2
                @addloglikelihood -0.5 * x
            end
            model = addloglik_multiple()
            _, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = parameters(accs)
            x_val = pars[@vn(x)]
            @test logprior(accs) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(accs) ≈ -x_val^2 + -0.5 * x_val
        end

        @testset "combined with conditioning" begin
            @wren function addloglik_with_cond()
                x ~ Normal()
                y ~ Normal(x)
                @addloglikelihood -0.5 * x^2
            end
            model = addloglik_with_cond() | VarNamedTuple(; y=1.0)
            _, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = parameters(accs)
            x_val = pars[@vn(x)]
            @test logprior(accs) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(accs) ≈ logdensityof(Normal(x_val), 1.0) + -0.5 * x_val^2
        end

        @testset "AST node" begin
            @wren function addloglik_ast()
                x ~ Normal()
                @addloglikelihood name=:penalty -x^2
            end
            stmts = filter(x -> x isa Wren.AbstractWrenExpr, addloglik_ast().stmts)
            addloglik_stmt = stmts[2]
            @test addloglik_stmt isa Wren.AddLogLikelihood
            @test addloglik_stmt.name === :penalty
        end

        @testset "elided without LogLikelihood accumulator" begin
            @wren function addloglik_elided()
                x ~ Normal()
                @addloglikelihood sum(collect(range(-x, x; length=1000)))
            end
            source = FromParams(VarNamedTuple(; x=0.5))
            model = addloglik_elided()

            # Sanity check: the expression does allocate when LogLikelihood is present
            result_with = @b evaluate(
                $model,
                $source,
                OriginalSpace(),
                $(Wren.default_accumulators()),
            )
            @test !iszero(result_with.allocs)

            # But is elided when LogLikelihood is absent
            accs_no_ll = Wren.deleteacc!!(
                Wren.default_accumulators(),
                Val(Wren.LOG_LIKELIHOOD_ACCNAME),
            )
            result_without = @b evaluate($model, $source, OriginalSpace(), $accs_no_ll)
            @test iszero(result_without.allocs)
        end
    end

    @testset "return values" begin
        @testset "returning a tilde statement" begin
            @wren function explicit_return_sample()
                return x ~ Normal()
            end
            model = explicit_return_sample()
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            @test retval == parameters(accs)[@vn(x)]
        end

        @testset "bare return" begin
            @wren function bare_return()
                return
                error("unreachable")
            end
            retval, _ = evaluate(
                bare_return(),
                FromPrior(),
                OriginalSpace(),
                Wren.default_accumulators(),
            )
            @test retval === nothing
        end

        @testset "fallthrough / implicit return" begin
            @wren function fallthrough_sample()
                x ~ Normal()
            end
            model = fallthrough_sample()
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            @test retval == parameters(accs)[@vn(x)]

            @wren function fallthrough_value()
                x ~ Normal()
                2
            end
            model = fallthrough_value()
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            @test retval == 2
        end
    end

    @testset "control flow" begin
        @wren function control_flow(n)
            if n == 0
                return :empty
            elseif iseven(n)
                branch = :even
            else
                branch = :odd
            end

            xs = Vector{Float64}(undef, n)
            for i in 1:n
                xs[i] ~ Normal()
            end

            i = 1
            while i <= n
                @observe xs[i] ~ Normal()
                i += 1
            end

            return (branch, :finished)
            error("unreachable")
        end

        values = [0.25, -0.5]
        source = FromParams(VarNamedTuple(; xs=values))
        retval, accs =
            evaluate(control_flow(2), source, OriginalSpace(), Wren.default_accumulators())
        expected_logp = sum(logdensityof(Normal(), value) for value in values)

        @test retval == (:even, :finished)
        xs = parameters(accs).data.xs
        @test VarNames.densify!!(xs) == values
        @test logprior(accs) ≈ expected_logp
        @test loglikelihood(accs) ≈ expected_logp

        retval, accs = evaluate(
            control_flow(0),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        @test retval == :empty
        @test parameters(accs) == VarNamedTuple()
    end

    @testset "model redefinition" begin
        # Check that redefining a model with the same name uses the updated binding.
        @wren function redefined_model()
            x ~ Normal()
        end
        out1 = rand(redefined_model())
        @test only(keys(out1)) == @vn(x)

        @wren function redefined_model()
            y ~ Normal()
        end
        out2 = rand(redefined_model())
        @test only(keys(out2)) == @vn(y)
    end
end

end # module WrenSyntaxTests
