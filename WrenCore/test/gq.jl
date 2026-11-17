module WrenGQTests

@info "testing gq.jl"

using Distributions
using FlexiChains: FlexiChains, Parameter, VNChain
using LinearAlgebra
using StableRNGs
using Test
using WrenCore
using WrenCore: GeneratedQuantity, is_accumulating_gq

# Run `@wren` on a quoted function definition and return the error message it throws.
function wren_error(ex)
    try
        WrenCore._wren(ex, @__MODULE__)
        return nothing
    catch e
        return sprint(showerror, e)
    end
end

const NOT_TOP_LEVEL = "@gq must be used at the top level of the model body"

@wren function gq_demo(x)
    a ~ Normal()
    b ~ Normal()
    mu = a .+ b .* x
    @gq mu
    @gq mu1 = mu[1]
    @gq s = begin
        t = sum(mu)
        t / length(mu)
    end
    @gq r = let z = a^2
        z + b
    end
    y ~ MvNormal(mu, I)
end

@wren function gq_throws()
    a ~ Normal()
    @gq bad = error("generated quantity was evaluated")
end

@wren function gq_inner()
    z0 ~ Normal()
    @gq z = 2z0
end

@wren function gq_outer()
    a ~ gq_inner()
    b ~ gq_inner()
    @gq c = 1.0
end

@testset "generated quantities" begin
    @testset "lowering" begin
        function lower(expr)
            return only(
                filter(
                    e -> e isa WrenCore.AbstractWrenExpr,
                    WrenCore.to_wren_exprs(expr, @__MODULE__),
                ),
            )
        end
        shorthand = lower(:(@gq x))
        @test shorthand isa GeneratedQuantity
        @test shorthand.name == :x
        @test shorthand.expr == :x

        full = lower(:(@gq y = f(x)))
        @test full.name == :y
        @test WrenCore._wren_expr_to_display_expr(full.expr) == :(f(x))

        @test lower(:(@gq s = begin
            t = 1
            t
        end)) isa GeneratedQuantity
        @test lower(:(@gq s = let t = 1
            t
        end)) isa GeneratedQuantity

        @test_throws "the name must be a single symbol" lower(:(@gq x[1] = 1.0))
        @test_throws "the name must be a single symbol" lower(:(@gq x.a = 1.0))
        @test_throws "expected `@gq name = expr` or `@gq name`" lower(:(@gq x y))
        @test_throws "must not contain probabilistic statements" lower(
            :(@gq s = (x ~ Normal())),
        )
        @test_throws "or `return`" lower(:(@gq s = begin
            return 1.0
        end))
    end

    @testset "inside control flow" begin
        @wren function gq_nested(flag)
            a ~ Normal()
            for i in 1:3
                @gq in_for = a + i
            end
            if flag
                @gq in_if = 2a
            end
            k = 0
            while k < 2
                k += 1
                @gq in_while = k
            end
            let z = a^2
                @gq in_let = z
            end
            begin
                @gq in_begin = a
            end
            @gq reused = a + 1
            for i in 1:2
                reused += 1
            end
            @gq reused_after = reused
        end

        model = gq_nested(true)
        gqs = generated_quantities(model)
        a = gqs[@vn(in_begin)]
        @test gqs[@vn(in_for)] ≈ a + 3
        @test gqs[@vn(in_if)] ≈ 2a
        @test gqs[@vn(in_while)] == 2
        @test gqs[@vn(in_let)] ≈ a^2
        @test gqs[@vn(reused)] ≈ a + 1
        @test gqs[@vn(reused_after)] ≈ a + 3
        @test collect(keys(generated_quantities(model, SomeGQs((:in_if,))))) == [@vn(in_if)]
        @test !haskey(generated_quantities(gq_nested(false)), @vn(in_if))
    end

    @testset "invalid positions" begin
        @test occursin(
            "@gq is not supported inside @batch",
            wren_error(:(function f(N)
                @batch for i in 1:N
                    @gq x = i
                end
            end)),
        )
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            function g()
                @gq x = 1.0
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            g = () -> (@gq x = 1.0)
        end)))
        @test occursin(
            "@gq must be used as a statement",
            wren_error(:(function f()
                y = (@gq x = 1.0)
            end)),
        )
        @test_throws "@gq must be used inside" @eval @gq x = 1.0
    end

    @testset "values" begin
        x = [1.0, 2.0]
        result = simulate(gq_demo(x))
        params, gqs = parameters(result), generated_quantities(result)
        a, b = params[@vn(a)], params[@vn(b)]
        mu = a .+ b .* x
        @test gqs[@vn(mu)] ≈ mu
        @test gqs[@vn(mu1)] ≈ mu[1]
        @test gqs[@vn(s)] ≈ sum(mu) / 2
        @test gqs[@vn(r)] ≈ a^2 + b
        @test haskey(params, @vn(y))
    end

    @testset "later uses of GQs trigger evaluation, even without GQ acc" begin
        @wren function gq_bind_return()
            @gq m = 1.0
            return m
        end
        @test generated_quantities(gq_bind_return())[@vn(m)] == 1.0
        retval, _ = evaluate(
            gq_bind_return(),
            FromPrior(),
            OriginalSpace(),
            AccumulatorTuple(),
            EvalContext{false}(),
        )
        @test retval == 1.0

        @wren function gq_bind_likelihood(x, y)
            a ~ Normal()
            @gq mu = a .+ x
            @observe y ~ MvNormal(mu, I)
            @gq bad = error("generated quantity was evaluated")
        end
        model = gq_bind_likelihood([0.0, 1.0], [0.5, 1.5])
        _, accs = evaluate(
            model,
            FromPrior(),
            OriginalSpace(),
            AccumulatorTuple(ParamsAccumulator(), LogLikelihoodAccumulator()),
            EvalContext{false}(),
        )
        a = parameters(accs)[@vn(a)]
        @test loglikelihood(accs) ≈ logpdf(MvNormal(a .+ [0.0, 1.0], I), [0.5, 1.5])

        mu = generated_quantities(model, SomeGQs((:mu,)))[@vn(mu)]
        @test mu ≈ mu[1] .+ [0.0, 1.0]
    end

    @testset "unrequested quantities are not evaluated" begin
        model = gq_throws()
        _, accs = evaluate(
            model,
            FromPrior(),
            OriginalSpace(),
            AccumulatorTuple(ParamsAccumulator()),
            EvalContext{false}(),
        )
        @test haskey(parameters(accs), @vn(a))
        @test isempty(generated_quantities(model, SomeGQs((:a,))))
        @test_throws "generated quantity was evaluated" simulate(model)
    end

    @testset "selection" begin
        model = gq_demo([1.0, 2.0])
        @test collect(keys(generated_quantities(model, SomeGQs((:s,))))) == [@vn(s)]
        @test Set(keys(generated_quantities(model, SomeGQs((:mu, :r))))) ==
              Set([@vn(mu), @vn(r)])
        @test isempty(generated_quantities(model, NoGQs()))
        @test isempty(generated_quantities(model, SomeGQs(())))
        @test length(keys(generated_quantities(model))) == 4

        accs = AccumulatorTuple(
            ParamsAccumulator(),
            GeneratedQuantityAccumulator(SomeGQs((:a,))),
        )
        @test is_accumulating_gq(accs, Val(:a))
        @test !is_accumulating_gq(accs, Val(:b))
        @test !is_accumulating_gq(AccumulatorTuple(ParamsAccumulator()), Val(:a))
        @test is_accumulating_gq(AccumulatorTuple(GeneratedQuantityAccumulator()), Val(:b))
    end

    @testset "submodels" begin
        model = gq_outer()
        all_gqs = generated_quantities(model)
        @test haskey(all_gqs, @vn(a.z))
        @test haskey(all_gqs, @vn(b.z))
        @test haskey(all_gqs, @vn(c))

        a_gqs = generated_quantities(model, SomeGQs((:a,)))
        @test haskey(a_gqs, @vn(a.z))
        @test !haskey(a_gqs, @vn(b))
        @test !haskey(a_gqs, @vn(c))

        b_gqs = generated_quantities(model, SomeGQs((:b,)))
        @test !haskey(b_gqs, @vn(a))
        @test haskey(b_gqs, @vn(b.z))

        result = simulate(model)
        @test generated_quantities(result)[@vn(a.z)] ≈ 2 * parameters(result)[@vn(a.z0)]

        scoped = @inferred WrenCore.scope(
            GeneratedQuantityAccumulator(SomeGQs((:a,))),
            @vn(a.b[1]),
        )
        @test scoped isa GeneratedQuantityAccumulator{AllGQs}
        scoped =
            @inferred WrenCore.scope(GeneratedQuantityAccumulator(SomeGQs((:a,))), @vn(b))
        @test scoped isa GeneratedQuantityAccumulator{NoGQs}
    end

    @testset "submodel result used only in a generated quantity" begin
        @wren function gq_uses_submodel()
            x ~ gq_inner()
            @gq doubled = 2 * x[@vn(z0)]
        end
        result = simulate(gq_uses_submodel())
        @test generated_quantities(result)[@vn(doubled)] ≈ 2 * parameters(result)[@vn(x.z0)]
    end

    @testset "entry points" begin
        model = gq_demo([1.0, 2.0])
        result = simulate(model)
        @test Set(keys(generated_quantities(result))) ==
              Set([@vn(mu), @vn(mu1), @vn(s), @vn(r)])
        @test haskey(parameters(result), @vn(a))
        @test !haskey(parameters(result), @vn(mu))
        @test !haskey(parameters(model), @vn(mu))
        @test isempty(generated_quantities(model, NoGQs()))

        @test Set(keys(generated_quantities(model))) ==
              Set([@vn(mu), @vn(mu1), @vn(s), @vn(r)])
        @test collect(keys(generated_quantities(model, SomeGQs((:s,))))) == [@vn(s)]
        @test isempty(generated_quantities(model, FromPrior(), NoGQs()))
        latents = VarNamedTuple(; a=1.0, b=2.0, y=[0.0, 0.0])
        gqs = generated_quantities(model, latents)
        @test gqs[@vn(mu)] ≈ [3.0, 5.0]
        @test collect(keys(generated_quantities(model, latents, SomeGQs((:r,))))) ==
              [@vn(r)]
    end

    @testset "Wren keeps GQs and parameters separate" begin
        # GQs and params do get smushed together when creating FlexiChains, but as far as
        # WrenCore is concerned, they're stored separately.
        @wren function gq_clash()
            a ~ Uniform(0.0, 1.0)
            @gq a = 100.0
        end
        res = simulate(gq_clash())
        @test parameters(res)[@vn(a)] < 1.0
        @test generated_quantities(res)[@vn(a)] == 100.0

        @wren function gq_clash_indexed()
            a = zeros(2)
            a[1] ~ Uniform(0.0, 1.0)
            @gq a = 100.0
        end
        res = simulate(gq_clash_indexed())
        @test parameters(res)[@vn(a[1])] < 1.0
        @test generated_quantities(res)[@vn(a)] == 100.0
    end

    @testset "generated quantities used only by other generated quantities" begin
        @wren function gq_chain()
            a ~ Normal()
            @gq b = error("b was evaluated")
            @gq c = b
        end
        model = gq_chain()
        @test isempty(generated_quantities(model, NoGQs()))
        @test_throws "b was evaluated" generated_quantities(model, AllGQs())
        @test_throws "b was evaluated" generated_quantities(model, SomeGQs((:c,)))
        @test_throws "b was evaluated" generated_quantities(model, SomeGQs((:b,)))

        @wren function gq_transitive()
            a ~ Normal()
            @gq b = a + 1
            @gq c = b + 1
            @gq d = c + 1
        end
        model = gq_transitive()
        d_only = generated_quantities(model, SomeGQs((:d,)))
        @test collect(keys(d_only)) == [@vn(d)]
        all_gqs = generated_quantities(model)
        @test all_gqs[@vn(d)] ≈ all_gqs[@vn(b)] + 2
    end

    @testset "type stability" begin
        @inferred simulate(gq_demo([1.0, 2.0]))
        @inferred simulate(gq_outer())
    end
end

@wren function gq_sub()
    z0 ~ Normal()
    @gq z = 2z0
end

@wren function sub_without_gqs()
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
    @wren function gq_model()
        x ~ Normal()
        a ~ gq_sub()
        @gq x2 = x^2
        @gq xp1 = x + 1
        y ~ Normal(x)
    end

    @wren function gq_model_without_gqs()
        x ~ Normal()
        a ~ sub_without_gqs()
        y ~ Normal(x)
    end

    @wren function gq_model_without_submodel()
        x ~ Normal()
        y ~ Normal(x)
    end

    model = gq_model()
    chn = prior_chain(StableRNG(468), gq_model_without_gqs(), 10, 2)

    gqs_all = generated_quantities(model, chn)
    gqs_only = generated_quantities(model, chn; gqs=SomeGQs((:x2, :a)))

    @testset "keys" begin
        @test parameter_keys(gqs_all) == Set([@vn(x2), @vn(xp1), @vn(a.z)])
        @test parameter_keys(gqs_only) == Set([@vn(x2), @vn(a.z)])
        @test parameter_keys(generated_quantities(model, chn; gqs=AllGQs())) ==
              parameter_keys(gqs_all)
        @test isempty(parameter_keys(generated_quantities(model, chn; gqs=NoGQs())))
    end

    @testset "forms agree on shared keys" begin
        for vn in parameter_keys(gqs_only)
            @test gqs_only[vn] == gqs_all[vn]
        end
    end

    @testset "values match chain parameters" begin
        @test gqs_all[@vn(x2)] == chn[@vn(x)] .^ 2
        @test gqs_all[@vn(xp1)] == chn[@vn(x)] .+ 1
        @test gqs_all[@vn(a.z)] == 2 .* chn[@vn(a.z0)]
    end

    @testset "indices are preserved" begin
        sub_chn = chn[iter=6:10]
        gqs = generated_quantities(model, sub_chn)
        @test FlexiChains.iter_indices(gqs) == FlexiChains.iter_indices(sub_chn)
        @test FlexiChains.chain_indices(gqs) == FlexiChains.chain_indices(chn)
        @test FlexiChains.nchains(gqs) == 2
    end

    @testset "missing parameters are drawn from the prior with rng" begin
        chn_without_a = prior_chain(StableRNG(468), gq_model_without_submodel(), 10, 2)
        gqs_a = generated_quantities(model, chn_without_a; rng=StableRNG(1))
        gqs_b = generated_quantities(model, chn_without_a; rng=StableRNG(1))
        gqs_c = generated_quantities(model, chn_without_a; rng=StableRNG(2))
        @test parameter_keys(gqs_a) == Set([@vn(x2), @vn(xp1), @vn(a.z)])
        @test gqs_a[@vn(x2)] == chn_without_a[@vn(x)] .^ 2
        @test gqs_a[@vn(a.z)] == gqs_b[@vn(a.z)]
        @test gqs_a[@vn(a.z)] != gqs_c[@vn(a.z)]
    end

    @testset "simulate on a chain holds generated quantities" begin
        simulated = simulate(model, chn; rng=StableRNG(468))
        @test issubset(Set([@vn(x2), @vn(xp1), @vn(a.z)]), parameter_keys(simulated))
        @test simulated[@vn(x2)] == chn[@vn(x)] .^ 2

        simulated_x2_only = simulate(model, chn; rng=StableRNG(468), gqs=SomeGQs((:x2,)))
        @test @vn(x2) in parameter_keys(simulated_x2_only)
        @test !(@vn(xp1) in parameter_keys(simulated_x2_only))
        @test !(@vn(a.z) in parameter_keys(simulated_x2_only))
    end

    @testset "parameters take precedence over generated quantities" begin
        @wren function chain_clash()
            a ~ Uniform(0.0, 1.0)
            @gq a = 100.0
        end
        result = simulate(chain_clash())
        clash_chn = WrenCore.to_flexichain(reshape([result], 1, 1))
        @test parameter_keys(clash_chn) == Set([@vn(a)])
        @test only(clash_chn[@vn(a)]) == parameters(result)[@vn(a)]

        @wren function chain_clash_indexed()
            a = zeros(2)
            a[1] ~ Uniform(0.0, 1.0)
            @gq a = 100.0
        end
        result = simulate(chain_clash_indexed())
        clash_chn = WrenCore.to_flexichain(reshape([result], 1, 1))
        @test parameter_keys(clash_chn) == Set([@vn(a[1])])
        @test only(clash_chn[@vn(a[1])]) == parameters(result)[@vn(a[1])]
    end
end

end # module WrenGQTests
