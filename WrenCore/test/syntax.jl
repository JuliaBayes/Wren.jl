module WrenSyntaxTests

@info "testing syntax.jl"

using WrenCore
using Test
using Distributions
using DensityInterface: logdensityof
using Chairmarks
using VarNames

macro preserve(expr)
    return esc(expr)
end

struct SiteNames{T} <: AbstractAccumulator
    names::T
end
WrenCore.accumulator_name(::Type{<:SiteNames}) = :sites
WrenCore.reset(::SiteNames) = SiteNames(())
Base.copy(acc::SiteNames) = acc
record_site(acc::SiteNames, ::Val{name}) where {name} = SiteNames((acc.names..., name))
WrenCore.accumulate_assume!!(acc::SiteNames, args...) = record_site(acc, last(args))
WrenCore.accumulate_observe!!(acc::SiteNames, args...) = record_site(acc, last(args))

@testset "syntax.jl" begin
    @testset "named sites" begin
        @wren function named_sites()
            @sample name=:prior x ~ Normal()
            y = 0.0
            @observe name=:likelihood y ~ Normal(x)
            @sample name=:outer_prior @iid a ~ Normal() size=2
            @iid size=2 @sample name=:inner_prior b ~ Normal()
            @observe name=:outer_likelihood @iid a ~ Normal()
            @iid @observe name=:inner_likelihood b ~ Normal()
            @sample @iid c ~ Normal() size=2 name=:prior
            @observe @iid c ~ Normal() name=:trailing_likelihood
            return x
        end
        values = VarNamedTuple(; x=0.5, a=[0.1, 0.2], b=[0.3, 0.4], c=[0.5, 0.6])
        _, accs = evaluate(
            named_sites(),
            FromValues(values),
            OriginalSpace(),
            AccumulatorTuple(
                ParamsAccumulator(),
                LogPriorAccumulator(),
                LogLikelihoodAccumulator(),
                SiteNames(()),
            ),
            EvalContext{false}(),
        )
        @test parameters(accs) == values
        @test accs[:sites].names == (
            :prior,
            :likelihood,
            :outer_prior,
            :inner_prior,
            :outer_likelihood,
            :inner_likelihood,
            :prior,
            :trailing_likelihood,
        )
        vm = VecModel(named_sites(), nothing; latent_source=FromValues(values))
        x = vector_parameters(vm, values)
        @test lp(vm, x) ≈ logjoint(accs)
    end

    @testset "basic models" begin
        @testset "1x sample" begin
            @wren function simple_normal()
                x ~ Normal()
                return nothing
            end
            model = simple_normal()
            @test model.name === :simple_normal
            result = simulate(model, FromPrior())
            pars = parameters(result)
            @test return_value(result) === nothing
            @test pars isa VarNamedTuple
            @test only(keys(pars)) == @vn(x)
            @test logprior(result) ≈ logdensityof(Normal(), pars[@vn(x)])
            @test iszero(loglikelihood(result))
        end

        @testset "1x sample with model argument" begin
            @wren function sample_with_arg(m)
                x ~ Normal(m[1])
                return nothing
            end
            d = [100.0]
            model = sample_with_arg(d)
            result = simulate(model, FromPrior())
            @test return_value(result) === nothing
            @test parameters(result) isa VarNamedTuple
            @test logprior(result) ≈ logdensityof(Normal(d[1]), parameters(result)[@vn(x)])
            @test iszero(loglikelihood(result))
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
            result = simulate(model, FromPrior())
            pars = parameters(result)
            @test return_value(result) === nothing
            @test pars isa VarNamedTuple
            @test collect(keys(pars)) == [@vn(x), @vn(y)]
            @test logprior(result) ≈
                  logdensityof(Normal(), pars[@vn(x)]) +
                  logdensityof(Normal(pars[@vn(x)]), pars[@vn(y)])
            @test iszero(loglikelihood(result))
        end

        @testset "after conditioning" begin
            conditioned_model = x_and_y() | (; y=1.0)
            @test conditioned_model.name === :x_and_y # shouldn't be changed.
            result = simulate(conditioned_model, FromPrior())
            pars = parameters(result)
            @test return_value(result) === nothing
            @test pars isa VarNamedTuple
            @test collect(keys(pars)) == [@vn(x)]
            @test logprior(result) ≈ logdensityof(Normal(), pars[@vn(x)])
            @test loglikelihood(result) ≈ logdensityof(Normal(pars[@vn(x)]), 1.0)
        end

        @testset "conditioning an argument replaces it" begin
            @wren function read_before_tilde(y)
                before = copy(y)
                y[1] ~ Normal()
                return before
            end
            model = read_before_tilde([0.0]) | (; y=[1.0])
            result = simulate(model, FromPrior())
            @test return_value(result) == [1.0]
            @test loglikelihood(result) ≈ logdensityof(Normal(), 1.0)
            params = VarNamedTuple(y=[2.0])
            @test logjoint(decondition(model, :y), params) ≈ logdensityof(Normal(), 2.0)
            @test logjoint(decondition(model), params) ≈ logdensityof(Normal(), 2.0)
        end

        @testset "a definition in the body is a template until its tilde" begin
            @wren function template_before_tilde()
                y = zeros(2)
                before = copy(y)
                for i in eachindex(y)
                    y[i] ~ Normal()
                end
                return (before, y)
            end
            model = template_before_tilde() | (; y=[1.0, 2.0])
            (before, after) = return_value(model, FromPrior())
            @test before == zeros(2)
            @test after == [1.0, 2.0]
        end

        @testset "whole structured observation" begin
            @wren function observed_fields(y)
                y.a ~ Normal()
                y.b ~ Normal(y.a)
                return y
            end
            data = (a=0.5, b=1.0)
            result = simulate(observed_fields((a=0.0, b=0.0)) | (y=data,), FromPrior())
            @test return_value(result) == data
            @test loglikelihood(result) ≈
                  logdensityof(Normal(), data.a) + logdensityof(Normal(data.a), data.b)
            # Removing a field must not make part of an ordinary observation latent.
            model = observed_fields((a=0.0, b=0.0)) | (y=data,)
            @test_throws ErrorException logjoint(
                decondition(model, @vn(y.a)),
                VarNamedTuple(y=(a=2.0,)),
            )
        end

        @testset "whole-array conditioning" begin
            @wren function observe_loop(N)
                x ~ Normal()
                y = zeros(N)
                for i in 1:N
                    y[i] ~ Normal(x)
                end
            end
            data = [1.0, 2.0, 3.0]
            @test_throws ArgumentError decondition(observe_loop(3) | (; y=data), @vn(y[1]))
            result = @inferred simulate(observe_loop(3) | (; y=data), FromPrior())
            x = parameters(result)[@vn(x)]
            @test loglikelihood(result) ≈ sum(logdensityof.(Normal(x), data))
            # Indexing past the conditioned array is an error, not a new latent variable.
            @test_throws ErrorException simulate(observe_loop(4) | (; y=data), FromPrior())
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
        result = simulate(model, FromPrior())
        pars = parameters(result)
        @test return_value(result) === nothing
        @test pars isa VarNamedTuple
        @test collect(keys(pars)) == [@vn(x)]
        @test logprior(result) ≈ logdensityof(Normal(), pars[@vn(x)])
        @test loglikelihood(result) ≈ logdensityof(Normal(), pars[@vn(x)] + 1)
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
        result = simulate(with_macros(), params)
        @test logprior(result) ≈ sum(logdensityof.(Ref(Normal()), [0.5, 1.0, 2.0]))
        @test loglikelihood(result) ≈ logdensityof(Normal(), 0.25)
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

        result = simulate(nested_if(true), params)
        @test logprior(result) ≈ logdensityof(Normal(), 0.5)
        @test loglikelihood(result) ≈ logdensityof(Normal(), 0.5)

        result = simulate(nested_if(false), params)
        @test logprior(result) ≈ logdensityof(Normal(), 2.0)
        @test loglikelihood(result) ≈ logdensityof(Normal(), 2.0)
    end

    @testset "immutable struct" begin
        struct Foo
            a::Float64
        end
        @wren function immutable_struct()
            x = Foo(0.5)
            x.a ~ Normal()
        end
        result = simulate(immutable_struct(), FromPrior())
        pars = parameters(result)
        @test only(keys(pars)) == @vn(x.a)
        @test logprior(result) ≈ logdensityof(Normal(), pars[@vn(x.a)])
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

            stmt = only(
                filter(x -> x isa WrenCore.AbstractWrenExpr, tilde_value_position().stmts),
            )
            @test stmt isa WrenCore.WrenExpr
            @test stmt.head == :(=)
            @test stmt.args[1] == :retval
            @test stmt.args[2] isa WrenCore.Sample
            @test stmt.args[2].lhs == :x
            @test stmt.args[2].rhs isa WrenCore.WrenExpr
            @test stmt.args[2].rhs.head == :call
            @test stmt.args[2].rhs.args == [:parser_inner]
        end

        @testset "with distributions" begin
            @wren function distribution_tilde_value()
                retval = x ~ Normal()
                @observe retval ~ Normal()
            end

            x = 0.5
            result = simulate(distribution_tilde_value(), VarNamedTuple(; x))
            @test parameters(result) == VarNamedTuple(; x)
            @test logprior(result) ≈ logdensityof(Normal(), x)
            @test loglikelihood(result) ≈ logdensityof(Normal(), x)

            conditioned_model = distribution_tilde_value() | (; x)
            result = simulate(conditioned_model, FromPrior())
            @test parameters(result) == VarNamedTuple()
            @test iszero(logprior(result))
            @test loglikelihood(result) ≈ 2 * logdensityof(Normal(), x)
        end
    end


    @testset "@addloglikelihood" begin
        @testset "anonymous" begin
            @wren function addloglik_anon()
                x ~ Normal()
                @addloglikelihood -0.5 * x^2
            end
            model = addloglik_anon()
            result = simulate(model, FromPrior())
            pars = parameters(result)
            x_val = pars[@vn(x)]
            @test logprior(result) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(result) ≈ -0.5 * x_val^2
        end

        @testset "named" begin
            @wren function addloglik_named()
                x ~ Normal()
                @addloglikelihood name=:penalty -x^2
            end
            model = addloglik_named()
            result = simulate(model, FromPrior())
            pars = parameters(result)
            x_val = pars[@vn(x)]
            @test logprior(result) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(result) ≈ -x_val^2
        end

        @testset "multiple addloglikelihood statements" begin
            @wren function addloglik_multiple()
                x ~ Normal()
                @addloglikelihood -x^2
                @addloglikelihood -0.5 * x
            end
            model = addloglik_multiple()
            result = simulate(model, FromPrior())
            pars = parameters(result)
            x_val = pars[@vn(x)]
            @test logprior(result) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(result) ≈ -x_val^2 + -0.5 * x_val
        end

        @testset "combined with conditioning" begin
            @wren function addloglik_with_cond()
                x ~ Normal()
                y ~ Normal(x)
                @addloglikelihood -0.5 * x^2
            end
            model = addloglik_with_cond() | (; y=1.0)
            result = simulate(model, FromPrior())
            pars = parameters(result)
            x_val = pars[@vn(x)]
            @test logprior(result) ≈ logdensityof(Normal(), x_val)
            @test loglikelihood(result) ≈ logdensityof(Normal(x_val), 1.0) + -0.5 * x_val^2
        end

        @testset "AST node" begin
            @wren function addloglik_ast()
                x ~ Normal()
                @addloglikelihood name=:penalty -x^2
            end
            stmts = filter(x -> x isa WrenCore.AbstractWrenExpr, addloglik_ast().stmts)
            addloglik_stmt = stmts[2]
            @test addloglik_stmt isa WrenCore.AddLogLikelihood
            @test addloglik_stmt.name === :penalty
        end

        @testset "elided without LogLikelihood accumulator" begin
            @wren function addloglik_elided()
                x ~ Normal()
                @addloglikelihood sum(collect(range(-x, x; length=1000)))
            end
            source = FromValues(VarNamedTuple(; x=0.5))
            model = addloglik_elided()

            # Sanity check: the expression does allocate when LogLikelihood is present
            accs_with_ll = AccumulatorTuple(
                LogPriorAccumulator(),
                LogLikelihoodAccumulator(),
                ParamsAccumulator(),
            )
            result_with = @b evaluate(
                $model,
                $source,
                OriginalSpace(),
                $accs_with_ll,
                EvalContext{false}(),
            )
            @test !iszero(result_with.allocs)

            # But is elided when LogLikelihood is absent
            accs_no_ll = AccumulatorTuple(LogPriorAccumulator(), ParamsAccumulator())
            result_without = @b evaluate(
                $model,
                $source,
                OriginalSpace(),
                $accs_no_ll,
                EvalContext{false}(),
            )
            @test iszero(result_without.allocs)
        end
    end

    @testset "return values" begin
        @testset "returning a tilde statement" begin
            @wren function explicit_return_sample()
                return x ~ Normal()
            end
            model = explicit_return_sample()
            result = simulate(model, FromPrior())
            @test return_value(result) == parameters(result)[@vn(x)]
        end

        @testset "bare return" begin
            @wren function bare_return()
                return
                error("unreachable")
            end
            retval = return_value(bare_return(), FromPrior())
            @test retval === nothing
        end

        @testset "fallthrough / implicit return" begin
            @wren function fallthrough_sample()
                x ~ Normal()
            end
            model = fallthrough_sample()
            result = simulate(model, FromPrior())
            @test return_value(result) == parameters(result)[@vn(x)]

            @wren function fallthrough_value()
                x ~ Normal()
                2
            end
            model = fallthrough_value()
            retval = return_value(model, FromPrior())
            @test retval == 2

            @wren function fallthrough_observe(y)
                x ~ Normal()
                @observe y ~ Normal(x)
            end
            model = fallthrough_observe(1.5)
            result = simulate(model, FromPrior())
            @test return_value(result) == 1.5
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
        result = simulate(control_flow(2), VarNamedTuple(; xs=values))
        expected_logp = sum(logdensityof(Normal(), value) for value in values)

        @test return_value(result) == (:even, :finished)
        xs = parameters(result).data.xs
        @test VarNames.densify!!(xs) == values
        @test logprior(result) ≈ expected_logp
        @test loglikelihood(result) ≈ expected_logp

        result = simulate(control_flow(0), FromPrior())
        @test return_value(result) == :empty
        @test parameters(result) == VarNamedTuple()
    end

    @testset "model redefinition" begin
        # Check that redefining a model with the same name uses the updated binding.
        @wren function redefined_model()
            x ~ Normal()
        end
        out1 = parameters(redefined_model())
        @test only(keys(out1)) == @vn(x)

        @wren function redefined_model()
            y ~ Normal()
        end
        out2 = parameters(redefined_model())
        @test only(keys(out2)) == @vn(y)
    end

    @testset "short function definitions" begin
        @wren short_def(a) = x ~ Normal(a)
        res = simulate(short_def(1.0))
        @test logprior(res) ≈ logdensityof(Normal(1.0), parameters(res)[@vn(x)])
    end

    @testset "typed, default, and keyword model arguments" begin
        @wren function f(a::Real, b=2.0; c=3.0)
            x ~ Normal()
            return (a, b, c)
        end
        for (model, expected) in
            ((f(1.0), (a=1.0, b=2.0, c=3.0)), (f(1.0, 4.0; c=5.0), (a=1.0, b=4.0, c=5.0)))
            @test model.args == expected
            retval = return_value(model, FromPrior())
            @test retval == values(expected)
        end
    end
end

end # module WrenSyntaxTests
