module WrenTQTests

@info "testing tq.jl"

using Distributions
using FlexiChains: FlexiChains, Parameter, VNChain
using LinearAlgebra
using StableRNGs
using Test
using WrenCore
using WrenCore: TrackedQuantity, is_accumulating_tq

# Run `@wren` on a quoted function definition and return the error message it throws.
function wren_error(ex)
    try
        WrenCore._wren(ex, @__MODULE__)
        return nothing
    catch e
        return sprint(showerror, e)
    end
end

const NOT_TOP_LEVEL = "@track must be used at the top level of the model body"

@wren function tq_demo(x)
    a ~ Normal()
    b ~ Normal()
    mu = a .+ b .* x
    @track mu
    @track mu1 = mu[1]
    @track s = begin
        t = sum(mu)
        t / length(mu)
    end
    @track r = let z = a^2
        z + b
    end
    y ~ MvNormal(mu, I)
end

@wren function tq_throws()
    a ~ Normal()
    @track bad = error("tracked quantity was evaluated")
end

@wren function tq_inner()
    z0 ~ Normal()
    @track z = 2z0
end

@wren function tq_outer()
    a ~ tq_inner()
    b ~ tq_inner()
    @track c = 1.0
end

@testset "tracked quantities" begin
    @testset "lowering" begin
        function lower(expr)
            return only(
                filter(
                    e -> e isa WrenCore.AbstractWrenExpr,
                    WrenCore.to_wren_exprs(expr, @__MODULE__),
                ),
            )
        end
        shorthand = lower(:(@track x))
        @test shorthand isa TrackedQuantity
        @test shorthand.name == :x
        @test shorthand.expr == :x

        full = lower(:(@track y = f(x)))
        @test full.name == :y
        @test WrenCore._wren_expr_to_display_expr(full.expr) == :(f(x))

        @test lower(:(@track s = begin
            t = 1
            t
        end)) isa TrackedQuantity
        @test lower(:(@track s = let t = 1
            t
        end)) isa TrackedQuantity

        @test_throws "the name must be a single symbol" lower(:(@track x[1] = 1.0))
        @test_throws "the name must be a single symbol" lower(:(@track x.a = 1.0))
        @test_throws "expected `@track name = expr` or `@track name`" lower(:(@track x y))
        @test_throws "must not contain probabilistic statements" lower(
            :(@track s = (x ~ Normal())),
        )
        @test_throws "or `return`" lower(:(@track s = begin
            return 1.0
        end))
    end

    @testset "inside control flow" begin
        @wren function tq_nested(flag)
            a ~ Normal()
            for i in 1:3
                @track in_for = a + i
            end
            if flag
                @track in_if = 2a
            end
            k = 0
            while k < 2
                k += 1
                @track in_while = k
            end
            let z = a^2
                @track in_let = z
            end
            begin
                @track in_begin = a
            end
            @track reused = a + 1
            for i in 1:2
                reused += 1
            end
            @track reused_after = reused
        end

        model = tq_nested(true)
        tqs = tracked_quantities(model)
        a = tqs[@vn(in_begin)]
        @test tqs[@vn(in_for)] ≈ a + 3
        @test tqs[@vn(in_if)] ≈ 2a
        @test tqs[@vn(in_while)] == 2
        @test tqs[@vn(in_let)] ≈ a^2
        @test tqs[@vn(reused)] ≈ a + 1
        @test tqs[@vn(reused_after)] ≈ a + 3
        @test collect(keys(tracked_quantities(model, SomeQuantities((:in_if,))))) ==
              [@vn(in_if)]
        @test !haskey(tracked_quantities(tq_nested(false)), @vn(in_if))
    end

    @testset "invalid positions" begin
        @test occursin(
            "@track is not supported inside @batch",
            wren_error(:(function f(N)
                @batch for i in 1:N
                    @track x = i
                end
            end)),
        )
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            function g()
                @track x = 1.0
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            g = () -> (@track x = 1.0)
        end)))
        @test occursin(
            "@track must be used as a statement",
            wren_error(:(function f()
                y = (@track x = 1.0)
            end)),
        )
        @test_throws "@track must be used inside" @eval @track x = 1.0
    end

    @testset "values" begin
        x = [1.0, 2.0]
        result = simulate(tq_demo(x))
        params, tqs = parameters(result), tracked_quantities(result)
        a, b = params[@vn(a)], params[@vn(b)]
        mu = a .+ b .* x
        @test tqs[@vn(mu)] ≈ mu
        @test tqs[@vn(mu1)] ≈ mu[1]
        @test tqs[@vn(s)] ≈ sum(mu) / 2
        @test tqs[@vn(r)] ≈ a^2 + b
        @test haskey(params, @vn(y))
    end

    @testset "later uses of TQs trigger evaluation, even without TQ acc" begin
        @wren function tq_bind_return()
            @track m = 1.0
            return m
        end
        @test tracked_quantities(tq_bind_return())[@vn(m)] == 1.0
        retval, _ = evaluate(
            tq_bind_return(),
            FromPrior(),
            OriginalSpace(),
            AccumulatorTuple(),
            EvalContext{false}(),
        )
        @test retval == 1.0

        @wren function tq_bind_likelihood(x, y)
            a ~ Normal()
            @track mu = a .+ x
            @observe y ~ MvNormal(mu, I)
            @track bad = error("tracked quantity was evaluated")
        end
        model = tq_bind_likelihood([0.0, 1.0], [0.5, 1.5])
        _, accs = evaluate(
            model,
            FromPrior(),
            OriginalSpace(),
            AccumulatorTuple(ParamsAccumulator(), LogLikelihoodAccumulator()),
            EvalContext{false}(),
        )
        a = parameters(accs)[@vn(a)]
        @test loglikelihood(accs) ≈ logpdf(MvNormal(a .+ [0.0, 1.0], I), [0.5, 1.5])

        mu = tracked_quantities(model, SomeQuantities((:mu,)))[@vn(mu)]
        @test mu ≈ mu[1] .+ [0.0, 1.0]
    end

    @testset "unrequested quantities are not evaluated" begin
        model = tq_throws()
        _, accs = evaluate(
            model,
            FromPrior(),
            OriginalSpace(),
            AccumulatorTuple(ParamsAccumulator()),
            EvalContext{false}(),
        )
        @test haskey(parameters(accs), @vn(a))
        @test isempty(tracked_quantities(model, SomeQuantities((:a,))))
        @test_throws "tracked quantity was evaluated" simulate(model)
    end

    @testset "selection" begin
        model = tq_demo([1.0, 2.0])
        @test collect(keys(tracked_quantities(model, SomeQuantities((:s,))))) == [@vn(s)]
        @test Set(keys(tracked_quantities(model, SomeQuantities((:mu, :r))))) ==
              Set([@vn(mu), @vn(r)])
        @test isempty(tracked_quantities(model, NoQuantities()))
        @test isempty(tracked_quantities(model, SomeQuantities(())))
        @test length(keys(tracked_quantities(model))) == 4

        accs = AccumulatorTuple(
            ParamsAccumulator(),
            TrackedQuantityAccumulator(SomeQuantities((:a,))),
        )
        @test is_accumulating_tq(accs, Val(:a))
        @test !is_accumulating_tq(accs, Val(:b))
        @test !is_accumulating_tq(AccumulatorTuple(ParamsAccumulator()), Val(:a))
        @test is_accumulating_tq(AccumulatorTuple(TrackedQuantityAccumulator()), Val(:b))
    end

    @testset "submodels" begin
        model = tq_outer()
        all_tqs = tracked_quantities(model)
        @test haskey(all_tqs, @vn(a.z))
        @test haskey(all_tqs, @vn(b.z))
        @test haskey(all_tqs, @vn(c))

        a_tqs = tracked_quantities(model, SomeQuantities((:a,)))
        @test haskey(a_tqs, @vn(a.z))
        @test !haskey(a_tqs, @vn(b))
        @test !haskey(a_tqs, @vn(c))

        b_tqs = tracked_quantities(model, SomeQuantities((:b,)))
        @test !haskey(b_tqs, @vn(a))
        @test haskey(b_tqs, @vn(b.z))

        result = simulate(model)
        @test tracked_quantities(result)[@vn(a.z)] ≈ 2 * parameters(result)[@vn(a.z0)]

        scoped = @inferred WrenCore.scope(
            TrackedQuantityAccumulator(SomeQuantities((:a,))),
            @vn(a.b[1]),
        )
        @test scoped isa TrackedQuantityAccumulator{AllQuantities}
        scoped = @inferred WrenCore.scope(
            TrackedQuantityAccumulator(SomeQuantities((:a,))),
            @vn(b),
        )
        @test scoped isa TrackedQuantityAccumulator{NoQuantities}
    end

    @testset "submodel result used only in a tracked quantity" begin
        @wren function tq_uses_submodel()
            x ~ tq_inner()
            @track doubled = 2 * x[@vn(z0)]
        end
        result = simulate(tq_uses_submodel())
        @test tracked_quantities(result)[@vn(doubled)] ≈ 2 * parameters(result)[@vn(x.z0)]
    end

    @testset "entry points" begin
        model = tq_demo([1.0, 2.0])
        result = simulate(model)
        @test Set(keys(tracked_quantities(result))) ==
              Set([@vn(mu), @vn(mu1), @vn(s), @vn(r)])
        @test haskey(parameters(result), @vn(a))
        @test !haskey(parameters(result), @vn(mu))
        @test !haskey(parameters(model), @vn(mu))
        @test isempty(tracked_quantities(model, NoQuantities()))

        @test Set(keys(tracked_quantities(model))) ==
              Set([@vn(mu), @vn(mu1), @vn(s), @vn(r)])
        @test collect(keys(tracked_quantities(model, SomeQuantities((:s,))))) == [@vn(s)]
        @test isempty(tracked_quantities(model, FromPrior(), NoQuantities()))
        latents = VarNamedTuple(; a=1.0, b=2.0, y=[0.0, 0.0])
        tqs = tracked_quantities(model, latents)
        @test tqs[@vn(mu)] ≈ [3.0, 5.0]
        @test collect(keys(tracked_quantities(model, latents, SomeQuantities((:r,))))) ==
              [@vn(r)]
    end

    @testset "Wren keeps TQs and parameters separate" begin
        # TQs and params do get smushed together when creating FlexiChains, but as far as
        # WrenCore is concerned, they're stored separately.
        @wren function tq_clash()
            a ~ Uniform(0.0, 1.0)
            @track a = 100.0
        end
        res = simulate(tq_clash())
        @test parameters(res)[@vn(a)] < 1.0
        @test tracked_quantities(res)[@vn(a)] == 100.0

        @wren function tq_clash_indexed()
            a = zeros(2)
            a[1] ~ Uniform(0.0, 1.0)
            @track a = 100.0
        end
        res = simulate(tq_clash_indexed())
        @test parameters(res)[@vn(a[1])] < 1.0
        @test tracked_quantities(res)[@vn(a)] == 100.0
    end

    @testset "tracked quantities used only by other tracked quantities" begin
        @wren function tq_chain()
            a ~ Normal()
            @track b = error("b was evaluated")
            @track c = b
        end
        model = tq_chain()
        @test isempty(tracked_quantities(model, NoQuantities()))
        @test_throws "b was evaluated" tracked_quantities(model, AllQuantities())
        @test_throws "b was evaluated" tracked_quantities(model, SomeQuantities((:c,)))
        @test_throws "b was evaluated" tracked_quantities(model, SomeQuantities((:b,)))

        @wren function tq_transitive()
            a ~ Normal()
            @track b = a + 1
            @track c = b + 1
            @track d = c + 1
        end
        model = tq_transitive()
        d_only = tracked_quantities(model, SomeQuantities((:d,)))
        @test collect(keys(d_only)) == [@vn(d)]
        all_tqs = tracked_quantities(model)
        @test all_tqs[@vn(d)] ≈ all_tqs[@vn(b)] + 2
    end

    @testset "type stability" begin
        @inferred simulate(tq_demo([1.0, 2.0]))
        @inferred simulate(tq_outer())
    end
end

@wren function tq_sub()
    z0 ~ Normal()
    @track z = 2z0
end

@wren function sub_without_tqs()
    z0 ~ Normal()
end

function prior_chain(rng, model, niters, nchains)
    results = map(Iterators.product(1:niters, 1:nchains)) do _
        simulate(model, rng)
    end
    return WrenCore.to_flexichain(results)
end

parameter_keys(chain) = Set(FlexiChains.parameters(chain))

@testset "on chains" begin
    @wren function tq_model()
        x ~ Normal()
        a ~ tq_sub()
        @track x2 = x^2
        @track xp1 = x + 1
        y ~ Normal(x)
    end

    @wren function tq_model_without_tqs()
        x ~ Normal()
        a ~ sub_without_tqs()
        y ~ Normal(x)
    end

    @wren function tq_model_without_submodel()
        x ~ Normal()
        y ~ Normal(x)
    end

    model = tq_model()
    chn = prior_chain(StableRNG(468), tq_model_without_tqs(), 10, 2)

    tqs_all = tracked_quantities(model, chn)
    tqs_only = tracked_quantities(model, chn; track=SomeQuantities((:x2, :a)))

    @testset "keys" begin
        @test parameter_keys(tqs_all) == Set([@vn(x2), @vn(xp1), @vn(a.z)])
        @test parameter_keys(tqs_only) == Set([@vn(x2), @vn(a.z)])
        @test parameter_keys(tracked_quantities(model, chn; track=AllQuantities())) ==
              parameter_keys(tqs_all)
        @test isempty(parameter_keys(tracked_quantities(model, chn; track=NoQuantities())))
    end

    @testset "forms agree on shared keys" begin
        for vn in parameter_keys(tqs_only)
            @test tqs_only[vn] == tqs_all[vn]
        end
    end

    @testset "values match chain parameters" begin
        @test tqs_all[@vn(x2)] == chn[@vn(x)] .^ 2
        @test tqs_all[@vn(xp1)] == chn[@vn(x)] .+ 1
        @test tqs_all[@vn(a.z)] == 2 .* chn[@vn(a.z0)]
    end

    @testset "indices are preserved" begin
        sub_chn = chn[iter=6:10]
        tqs = tracked_quantities(model, sub_chn)
        @test FlexiChains.iter_indices(tqs) == FlexiChains.iter_indices(sub_chn)
        @test FlexiChains.chain_indices(tqs) == FlexiChains.chain_indices(chn)
        @test FlexiChains.nchains(tqs) == 2
    end

    @testset "missing parameters are drawn from the prior with rng" begin
        chn_without_a = prior_chain(StableRNG(468), tq_model_without_submodel(), 10, 2)
        tqs_a = tracked_quantities(model, chn_without_a; rng=StableRNG(1))
        tqs_b = tracked_quantities(model, chn_without_a; rng=StableRNG(1))
        tqs_c = tracked_quantities(model, chn_without_a; rng=StableRNG(2))
        @test parameter_keys(tqs_a) == Set([@vn(x2), @vn(xp1), @vn(a.z)])
        @test tqs_a[@vn(x2)] == chn_without_a[@vn(x)] .^ 2
        @test tqs_a[@vn(a.z)] == tqs_b[@vn(a.z)]
        @test tqs_a[@vn(a.z)] != tqs_c[@vn(a.z)]
    end

    @testset "simulate on a chain holds tracked quantities" begin
        simulated = simulate(model, chn; rng=StableRNG(468))
        @test issubset(Set([@vn(x2), @vn(xp1), @vn(a.z)]), parameter_keys(simulated))
        @test simulated[@vn(x2)] == chn[@vn(x)] .^ 2

        simulated_x2_only =
            simulate(model, chn; rng=StableRNG(468), track=SomeQuantities((:x2,)))
        @test @vn(x2) in parameter_keys(simulated_x2_only)
        @test !(@vn(xp1) in parameter_keys(simulated_x2_only))
        @test !(@vn(a.z) in parameter_keys(simulated_x2_only))
    end

    @testset "parameters take precedence over tracked quantities" begin
        @wren function chain_clash()
            a ~ Uniform(0.0, 1.0)
            @track a = 100.0
        end
        result = simulate(chain_clash())
        clash_chn = WrenCore.to_flexichain(reshape([result], 1, 1))
        @test parameter_keys(clash_chn) == Set([@vn(a)])
        @test only(clash_chn[@vn(a)]) == parameters(result)[@vn(a)]

        @wren function chain_clash_indexed()
            a = zeros(2)
            a[1] ~ Uniform(0.0, 1.0)
            @track a = 100.0
        end
        result = simulate(chain_clash_indexed())
        clash_chn = WrenCore.to_flexichain(reshape([result], 1, 1))
        @test parameter_keys(clash_chn) == Set([@vn(a[1])])
        @test only(clash_chn[@vn(a[1])]) == parameters(result)[@vn(a[1])]
    end
end

end # module WrenTQTests
