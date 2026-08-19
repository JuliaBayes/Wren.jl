module WrenSubmodelTests

using Wren
using Test
using Distributions
using DensityInterface: logdensityof
using VarNames

@testset "submodels" begin
    @testset "basic" begin
        @model function returning_inner()
            a ~ Normal()
            return 2a
        end
        @model function returning_outer(inner)
            inner_retval = x ~ inner
            # x is now the child params VarNamedTuple
            return (x, inner_retval + 1)
        end

        # With FromPrior
        retval, accs = evaluate(
            returning_outer(returning_inner()),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        out_pars = params(accs)
        a = out_pars[@vn(x.a)]
        @test retval == (VarNamedTuple(; a=a), 2a + 1)
        @test only(keys(out_pars)) == @vn(x.a)
        @test logprior(accs) ≈ logdensityof(Normal(), a)
        @test iszero(loglikelihood(accs))

        # With FromParams
        in_pars = @vnt begin
            x.a := 0.5
        end
        retval, accs = evaluate(
            returning_outer(returning_inner()),
            FromParams(in_pars),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        out_pars = params(accs)
        @test retval == (VarNamedTuple(; a=in_pars[@vn(x.a)]), 2 * in_pars[@vn(x.a)] + 1)
        @test only(keys(out_pars)) == @vn(x.a)
        @test logprior(accs) ≈ logdensityof(Normal(), in_pars[@vn(x.a)])
        @test iszero(loglikelihood(accs))
    end

    @testset "undefined namespace" begin
        @model function returning_inner()
            a ~ Normal()
            return 2a
        end
        @model function undefined_namespace_outer(inner)
            # Should work even if `x` is not defined yet
            retval = x ~ inner
            return retval
        end

        # With FromPrior
        retval, accs = evaluate(
            undefined_namespace_outer(returning_inner()),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        out_pars = params(accs)
        @test only(keys(out_pars)) == @vn(x.a)
        a = out_pars[@vn(x.a)]
        @test retval == 2a

        # With FromParams
        in_pars = @vnt begin
            x.a := 0.5
        end
        retval, accs = evaluate(
            undefined_namespace_outer(returning_inner()),
            FromParams(in_pars),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        out_pars = params(accs)
        @test only(keys(out_pars)) == @vn(x.a)
        @test retval == 2 * in_pars[@vn(x.a)]
    end

    @testset "indexed namespace uses the outer template" begin
        @model function indexed_inner()
            a ~ Normal()
        end

        @model function indexed_outer(inner)
            x = zeros(2, 2)
            first_retval = x[1] ~ inner
            second_retval = x[2, 2] ~ inner
            return (first_retval, second_retval)
        end

        # With FromPrior
        retval, accs = evaluate(
            indexed_outer(indexed_inner()),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        out_pars = params(accs)
        @test size(out_pars.data.x) == (2, 2) # tests that templated_setindex worked
        a1 = out_pars[@vn(x[1, 1].a)]
        a2 = out_pars[@vn(x[2, 2].a)]
        @test retval == (a1, a2)

        # With FromParams
        first_a = 0.25
        second_a = -0.5
        in_pars = @vnt begin
            @template x = zeros(2, 2)
            x[1].a := first_a
            x[2, 2].a := second_a
        end
        retval, accs = evaluate(
            indexed_outer(indexed_inner()),
            FromParams(in_pars),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        out_pars = params(accs)

        @test retval == (first_a, second_a)
        @test Set(keys(out_pars)) == Set([@vn(x[1, 1].a), @vn(x[2, 2].a)])
        @test out_pars[@vn(x[1].a)] == first_a
        @test out_pars[@vn(x[2, 2].a)] == second_a
        @test logprior(accs) ≈
              logdensityof(Normal(), first_a) + logdensityof(Normal(), second_a)
    end

    @testset "child assumptions and observations" begin
        @model function likelihood_inner()
            a ~ Normal()
            obs ~ Normal(a)
            return a
        end
        @model function likelihood_outer(inner)
            # This model doesn't have likelihood per se, it's just the outer
            # model that wraps a likelihood one
            inner_retval = x ~ inner
            b ~ Normal(inner_retval)
            return b
        end

        a = 0.4
        b = -0.2
        obs = 0.1
        in_pars = @vnt begin
            x.a := a
            b := b
        end
        retval, accs = evaluate(
            likelihood_outer(likelihood_inner() | VarNamedTuple(; obs=obs)),
            FromParams(in_pars),
            OriginalSpace(),
            Wren.default_accumulators(),
        )

        @test retval == b
        @test params(accs) == in_pars
        @test logprior(accs) ≈ logdensityof(Normal(), a) + logdensityof(Normal(a), b)
        @test loglikelihood(accs) ≈ logdensityof(Normal(a), obs)
    end

    @testset "empty submodel" begin
        @model function empty_inner()
            # no sampling statements
            return 42
        end
        @model function empty_outer(inner)
            # This model should also be empty as a result of that
            return x ~ inner
            # But because `x ~ inner` evaluates to the retval of inner, we should
            # also return 42 from this model
        end

        retval, accs = evaluate(
            empty_outer(empty_inner()),
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        @test retval == 42
        @test params(accs) == VarNamedTuple()
        @test iszero(logjoint(accs))
    end

    @testset "nested submodels" begin
        @model function returning_inner()
            a ~ Normal()
            return 2a
        end
        @model function middle_model(inner)
            retval = y ~ inner
            return retval + 1
        end
        @model function outer_model(inner)
            retval = x ~ inner
            return retval + 1
        end
        nested_model = outer_model(middle_model(returning_inner()))

        # With FromPrior
        retval, accs = evaluate(
            nested_model,
            FromPrior(),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        out_pars = params(accs)
        @test only(keys(out_pars)) == @vn(x.y.a)
        @test retval ≈ 2 * out_pars[@vn(x.y.a)] + 2

        # With FromParams
        a = 0.75
        in_pars = @vnt begin
            x.y.a := a
        end
        nested_model = outer_model(middle_model(returning_inner()))
        retval, accs = evaluate(
            nested_model,
            FromParams(in_pars),
            OriginalSpace(),
            Wren.default_accumulators(),
        )
        @test retval ≈ 2a + 2
        @test params(accs) == in_pars
        @test logprior(accs) ≈ logdensityof(Normal(), a)
    end
end

end # module WrenSubmodelTests
