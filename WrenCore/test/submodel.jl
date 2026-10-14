module WrenSubmodelTests

@info "testing submodel.jl"

using WrenCore
using Enzyme
using ForwardDiff
using Test
using Distributions
using VarNames

@wren function cached_inner()
    a ~ Normal()
end
@wren function cached_outer()
    x ~ cached_inner()
end

@testset "submodels" begin
    @testset "basic" begin
        @wren function returning_inner()
            a ~ Normal()
            return 2a
        end
        @wren function returning_outer(inner)
            inner_retval = x ~ inner
            # x is now the child params VarNamedTuple
            return (x, inner_retval + 1)
        end

        # With FromPrior
        retval, accs = @inferred evaluate(
            returning_outer(returning_inner()),
            FromPrior(),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        out_pars = parameters(accs)
        a = out_pars[@vn(x.a)]
        @test retval == (VarNamedTuple(; a=a), 2a + 1)
        @test only(keys(out_pars)) == @vn(x.a)
        @test logprior(accs) ≈ logdensityof(Normal(), a)
        @test iszero(loglikelihood(accs))

        # With FromValues
        in_pars = @vnt begin
            x.a := 0.5
        end
        retval, accs = @inferred evaluate(
            returning_outer(returning_inner()),
            FromValues(in_pars),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        out_pars = parameters(accs)
        @test retval == (VarNamedTuple(; a=in_pars[@vn(x.a)]), 2 * in_pars[@vn(x.a)] + 1)
        @test only(keys(out_pars)) == @vn(x.a)
        @test logprior(accs) ≈ logdensityof(Normal(), in_pars[@vn(x.a)])
        @test iszero(loglikelihood(accs))
    end

    @testset "reuse evaluation function" begin
        inner1 = cached_inner()
        inner2 = cached_inner()
        @test typeof(inner1.evaluation_fn) === typeof(inner2.evaluation_fn)
        @test typeof(inner1) === typeof(inner2)

        outer = cached_outer()
        rand(outer)
        @test typeof(cached_inner()) === typeof(inner1)
    end

    @testset "undefined namespace" begin
        @wren function returning_inner()
            a ~ Normal()
            return 2a
        end
        @wren function undefined_namespace_outer(inner)
            # Should work even if `x` is not defined yet
            retval = x ~ inner
            return retval
        end

        # With FromPrior
        retval, accs = @inferred evaluate(
            undefined_namespace_outer(returning_inner()),
            FromPrior(),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        out_pars = parameters(accs)
        @test only(keys(out_pars)) == @vn(x.a)
        a = out_pars[@vn(x.a)]
        @test retval == 2a

        # With FromValues
        in_pars = @vnt begin
            x.a := 0.5
        end
        retval, accs = @inferred evaluate(
            undefined_namespace_outer(returning_inner()),
            FromValues(in_pars),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        out_pars = parameters(accs)
        @test only(keys(out_pars)) == @vn(x.a)
        @test retval == 2 * in_pars[@vn(x.a)]
    end

    @testset "indexed namespace uses the outer template" begin
        @wren function indexed_inner()
            a ~ Normal()
        end

        @wren function indexed_outer(inner)
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
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        out_pars = parameters(accs)
        @test size(out_pars.data.x) == (2, 2) # tests that templated_setindex worked
        a1 = out_pars[@vn(x[1, 1].a)]
        a2 = out_pars[@vn(x[2, 2].a)]
        @test retval == (a1, a2)

        # With FromValues
        first_a = 0.25
        second_a = -0.5
        in_pars = @vnt begin
            @template x = zeros(2, 2)
            x[1].a := first_a
            x[2, 2].a := second_a
        end
        retval, accs = evaluate(
            indexed_outer(indexed_inner()),
            FromValues(in_pars),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        out_pars = parameters(accs)

        @test retval == (first_a, second_a)
        @test Set(keys(out_pars)) == Set([@vn(x[1, 1].a), @vn(x[2, 2].a)])
        @test out_pars[@vn(x[1].a)] == first_a
        @test out_pars[@vn(x[2, 2].a)] == second_a
        @test logprior(accs) ≈
              logdensityof(Normal(), first_a) + logdensityof(Normal(), second_a)
    end

    @testset "child assumptions and observations" begin
        @wren function likelihood_inner()
            a ~ Normal()
            obs ~ Normal(a)
            return a
        end
        @wren function likelihood_outer(inner)
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
        retval, accs = @inferred evaluate(
            likelihood_outer(likelihood_inner() | VarNamedTuple(; obs=obs)),
            FromValues(in_pars),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )

        @test retval == b
        @test parameters(accs) == in_pars
        @test logprior(accs) ≈ logdensityof(Normal(), a) + logdensityof(Normal(a), b)
        @test loglikelihood(accs) ≈ logdensityof(Normal(a), obs)

        # https://github.com/JuliaBayes/Wren.jl/issues/162
        # The parent's conditioned value reaches the child and overrides the child's own.
        parent_conditioned = @vnt begin
            x.obs := obs
        end
        for inner in (likelihood_inner(), likelihood_inner() | VarNamedTuple(; obs=0.9))
            _, accs = @inferred evaluate(
                likelihood_outer(inner) | parent_conditioned,
                FromValues(in_pars),
                OriginalSpace(),
                WrenCore.default_accumulators(),
                EvalContext{false}(),
            )
            @test parameters(accs) == in_pars
            @test loglikelihood(accs) ≈ logdensityof(Normal(a), obs)
        end
    end

    @testset "empty submodel" begin
        @wren function empty_inner()
            # no sampling statements
            return 42
        end
        @wren function empty_outer(inner)
            # This model should also be empty as a result of that
            return x ~ inner
            # But because `x ~ inner` evaluates to the retval of inner, we should
            # also return 42 from this model
        end

        retval, accs = @inferred evaluate(
            empty_outer(empty_inner()),
            FromPrior(),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        @test retval == 42
        @test parameters(accs) == VarNamedTuple()
        @test iszero(logjoint(accs))

        @test iszero(logjoint(empty_outer(empty_inner()), VarNamedTuple()))

        @wren function empty_outer_with_latent(inner)
            mu = child ~ inner
            x ~ Normal(mu)
        end
        @test logjoint(empty_outer_with_latent(empty_inner()), VarNamedTuple(x=43.0)) ≈
              logdensityof(Normal(42), 43.0)
        @test_throws ErrorException logjoint(cached_outer(), VarNamedTuple())
    end

    @testset "nested submodels" begin
        @wren function returning_inner()
            a ~ Normal()
            return 2a
        end
        @wren function middle_model(inner)
            retval = y ~ inner
            return retval + 1
        end
        @wren function outer_model(inner)
            retval = x ~ inner
            return retval + 1
        end
        nested_model = outer_model(middle_model(returning_inner()))

        # With FromPrior
        retval, accs = @inferred evaluate(
            nested_model,
            FromPrior(),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        out_pars = parameters(accs)
        @test only(keys(out_pars)) == @vn(x.y.a)
        @test retval ≈ 2 * out_pars[@vn(x.y.a)] + 2

        # With FromValues
        a = 0.75
        in_pars = @vnt begin
            x.y.a := a
        end
        nested_model = outer_model(middle_model(returning_inner()))
        # TODO: not type stable, needs #5 before we can add inferred
        @test_broken false
        retval, accs = evaluate(
            nested_model,
            FromValues(in_pars),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        @test retval ≈ 2a + 2
        @test parameters(accs) == in_pars
        @test logprior(accs) ≈ logdensityof(Normal(), a)
    end

    @testset "vectorisation" begin
        @testset "basic" begin
            @wren function innerv()
                x ~ Normal()
            end
            @wren function outerv(sub)
                a ~ sub
            end
            vmodel = VecModel(outerv(innerv()), AutoEnzyme(; mode=Reverse))

            r = rand(vmodel)
            @test r isa AbstractVector{<:Real}
            @test length(r) == 1
            @test lp(vmodel, r) isa Real
            @test last(lp_and_grad(vmodel, r)) isa AbstractVector{<:Real}
        end

        @testset "no fast path sitenames in submodel" begin
            @wren function innerv2(N)
                # No unique sitenames in this model
                x = zeros(N)
                for i in 1:N
                    x[i] ~ Normal()
                end
            end
            @wren function outerv2(sub)
                a ~ sub
            end
            N = 5
            vmodel = VecModel(outerv2(innerv2(N)), AutoEnzyme(; mode=Reverse))

            # Check that there's an entry for the submodel even though it has
            # no unique sitenames
            @test haskey(vmodel.vecinfo.sitename_to_rat, @vn(a))

            r = rand(vmodel)
            @test r isa AbstractVector{<:Real}
            @test length(r) == N
            @test lp(vmodel, r) isa Real
            @test last(lp_and_grad(vmodel, r)) isa AbstractVector{<:Real}
        end

        @testset "repeated submodel" begin
            @wren function innerv3()
                x ~ Normal()
            end
            @wren function outerv3(sub)
                a = zeros(5)
                for i in 1:5
                    a[i] ~ sub
                end
            end
            vmodel = VecModel(outerv3(innerv3()), AutoForwardDiff())
            # Enzyme fails (https://github.com/JuliaBayes/Wren.jl/issues/10 ???)
            # vmodel = VecModel(outerv3(innerv3()), AutoEnzyme(; mode=Reverse))
            @test_broken false

            @test all(i -> haskey(vmodel.vecinfo.sitename_to_rat, @vn(a[i])), 1:5)

            r = rand(vmodel)
            @test r isa AbstractVector{<:Real}
            @test length(r) == 5
            @test lp(vmodel, r) isa Real
            @test last(lp_and_grad(vmodel, r)) isa AbstractVector{<:Real}
        end
    end
end

end # module WrenSubmodelTests
