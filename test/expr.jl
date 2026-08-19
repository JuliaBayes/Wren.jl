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
            @model function simple_normal()
                x ~ Normal()
                return nothing
            end
            model = simple_normal()
            @test model.name === :simple_normal
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = params(accs)
            @test retval === nothing
            @test pars isa VarNamedTuple
            @test only(keys(pars)) == @vn(x)
            @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x)])
            @test iszero(loglikelihood(accs))
        end

        @testset "1x sample with model argument" begin
            @model function sample_with_arg(m)
                x ~ Normal(m[1])
                return nothing
            end
            d = [100.0]
            model = sample_with_arg(d)
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            @test retval === nothing
            @test params(accs) isa VarNamedTuple
            @test logprior(accs) ≈ logdensityof(Normal(d[1]), params(accs)[@vn(x)])
            @test iszero(loglikelihood(accs))
        end
    end

    @testset "conditioning" begin
        @model function x_and_y()
            x ~ Normal()
            y ~ Normal(x)
            return nothing
        end

        @testset "before conditioning" begin
            model = x_and_y()
            @test model.name === :x_and_y
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            pars = params(accs)
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
            pars = params(accs)
            @test retval === nothing
            @test pars isa VarNamedTuple
            @test collect(keys(pars)) == [@vn(x)]
            @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x)])
            @test loglikelihood(accs) ≈ logdensityof(Normal(pars[@vn(x)]), 1.0)
        end
    end

    @testset "observe macro" begin
        @model function observe_macro()
            x ~ Normal()
            y = x + 1
            @observe y ~ Normal()
            return nothing
        end
        model = observe_macro()
        @test model.name === :observe_macro
        retval, accs =
            evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
        pars = params(accs)
        @test retval === nothing
        @test pars isa VarNamedTuple
        @test collect(keys(pars)) == [@vn(x)]
        @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x)])
        @test loglikelihood(accs) ≈ logdensityof(Normal(), pars[@vn(x)] + 1)
    end

    @testset "macros inside models" begin
        @model function with_macros()
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
        @model function nested_if(flag)
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
        @model function immutable_struct()
            x = Foo(0.5)
            x.a ~ Normal()
        end
        _, accs = evaluate(
            immutable_struct(),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        pars = params(accs)
        @test only(keys(pars)) == @vn(x.a)
        @test logprior(accs) ≈ logdensityof(Normal(), pars[@vn(x.a)])
    end

    @testset "a = b ~ c syntax" begin
        @testset "macro parses it correctly" begin
            # Essentially we want `a = b ~ c` to be parsed as `a = Sample(b, c)`.
            @model function parser_inner()
                return 1
            end

            @model function tilde_value_position()
                retval = x ~ parser_inner()
            end

            stmt = only(filter(x -> x isa Wren.AbstractWrenExpr, tilde_value_position().stmts))
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
            @model function distribution_tilde_value()
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
            @test params(accs) == VarNamedTuple(; x)
            @test logprior(accs) ≈ logdensityof(Normal(), x)
            @test loglikelihood(accs) ≈ logdensityof(Normal(), x)

            conditioned_model = distribution_tilde_value() | VarNamedTuple(; x)
            _, accs = evaluate(
                conditioned_model,
                FromPrior(),
                OriginalSpace(),
                Wren.default_accumulators(),
            )
            @test params(accs) == VarNamedTuple()
            @test iszero(logprior(accs))
            @test loglikelihood(accs) ≈ 2 * logdensityof(Normal(), x)
        end
    end


    @testset "return values" begin
        @testset "returning a tilde statement" begin
            @model function explicit_return_sample()
                return x ~ Normal()
            end
            model = explicit_return_sample()
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            @test retval == params(accs)[@vn(x)]
        end

        @testset "bare return" begin
            @model function bare_return(events)
                # Mutate `events` to test that the return statement does indeed exit early
                push!(events, :before)
                return
                push!(events, :after)
            end
            events = Symbol[]
            retval, _ = evaluate(
                bare_return(events),
                FromPrior(),
                OriginalSpace(),
                Wren.default_accumulators(),
            )
            @test retval === nothing
            @test events == [:before]
        end

        @testset "fallthrough / implicit return" begin
            @model function fallthrough_sample()
                x ~ Normal()
            end
            model = fallthrough_sample()
            retval, accs =
                evaluate(model, FromPrior(), OriginalSpace(), Wren.default_accumulators())
            @test retval == params(accs)[@vn(x)]

            @model function fallthrough_value()
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
        @model function control_flow(n, events)
            if n == 0
                return push!(events, :empty)
            elseif iseven(n)
                push!(events, :even)
            else
                push!(events, :odd)
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

            push!(events, :finished)
            return
            push!(events, :unreachable)
        end

        values = [0.25, -0.5]
        source = FromParams(VarNamedTuple(; xs=values))
        events = Symbol[]
        _, accs = evaluate(
            control_flow(2, events),
            source,
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        expected_logp = sum(logdensityof(Normal(), value) for value in values)

        @test events == [:even, :finished]
        xs = params(accs).data.xs
        @test VarNames.densify!!(xs) == values
        @test logprior(accs) ≈ expected_logp
        @test loglikelihood(accs) ≈ expected_logp

        events = Symbol[]
        _, accs = evaluate(
            control_flow(0, events),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        @test events == [:empty]
        @test params(accs) == VarNamedTuple()
    end

end

end # module WrenSyntaxTests
