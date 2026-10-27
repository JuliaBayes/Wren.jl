module WrenGQTests

@info "testing gq.jl"

using Distributions
using LinearAlgebra
using Test
using WrenCore
using WrenCore: GeneratedQuantity, requests_gq

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

@noinline gq_helper(x) = sum(x) + 1.0

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

@wren function gq_pruned(x)
    a ~ Normal()
    @gq h = gq_helper(a .+ x)
end

function gq_values(model, acc)
    _, accs = evaluate(
        model,
        FromPrior(),
        OriginalSpace(),
        AccumulatorTuple(acc),
        EvalContext{false}(),
    )
    return generated_quantities(accs)
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

    @testset "only at the top level" begin
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            for i in 1:2
                @gq x = i
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            if true
                @gq x = 1.0
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            while false
                @gq x = 1.0
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            let
                @gq x = 1.0
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            begin
                @gq x = 1.0
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f(N)
            @batch for i in 1:N
                @gq x = i
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            function g()
                @gq x = 1.0
            end
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            g = () -> (@gq x = 1.0)
        end)))
        @test occursin(NOT_TOP_LEVEL, wren_error(:(function f()
            y = (@gq x = 1.0)
        end)))
        @test_throws "@gq must be used at the top level of a @wren model body" @eval @gq x =
            1.0
    end

    @testset "values" begin
        x = [1.0, 2.0]
        params = rand(gq_demo(x))
        a, b = params[@vn(a)], params[@vn(b)]
        mu = a .+ b .* x
        @test params[@vn(mu)] ≈ mu
        @test params[@vn(mu1)] ≈ mu[1]
        @test params[@vn(s)] ≈ sum(mu) / 2
        @test params[@vn(r)] ≈ a^2 + b
        @test haskey(params, @vn(y))
    end

    @testset "@gq does not bind its name" begin
        @wren function gq_nobind()
            @gq m = 1.0
            return m
        end
        @test_throws UndefVarError rand(gq_nobind())
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
        @test isempty(gq_values(model, GQAccumulator((:a,))))
        @test_throws "generated quantity was evaluated" rand(model)
    end

    @testset "selection" begin
        model = gq_demo([1.0, 2.0])
        @test collect(keys(gq_values(model, GQAccumulator((:s,))))) == [@vn(s)]
        @test Set(keys(gq_values(model, GQAccumulator((:mu, :r))))) ==
              Set([@vn(mu), @vn(r)])
        @test isempty(gq_values(model, GQAccumulator(())))
        @test length(keys(gq_values(model, GQAccumulator()))) == 4

        accs = AccumulatorTuple(ParamsAccumulator(), GQAccumulator((:a,)))
        @test requests_gq(accs, Val(:a))
        @test !requests_gq(accs, Val(:b))
        @test !requests_gq(AccumulatorTuple(ParamsAccumulator()), Val(:a))
        @test requests_gq(AccumulatorTuple(GQAccumulator()), Val(:b))
    end

    @testset "submodels" begin
        model = gq_outer()
        all_gqs = gq_values(model, GQAccumulator())
        @test haskey(all_gqs, @vn(a.z))
        @test haskey(all_gqs, @vn(b.z))
        @test haskey(all_gqs, @vn(c))

        a_gqs = gq_values(model, GQAccumulator((:a,)))
        @test haskey(a_gqs, @vn(a.z))
        @test !haskey(a_gqs, @vn(b))
        @test !haskey(a_gqs, @vn(c))

        b_gqs = gq_values(model, GQAccumulator((:b,)))
        @test !haskey(b_gqs, @vn(a))
        @test haskey(b_gqs, @vn(b.z))

        params = rand(model)
        @test params[@vn(a.z)] ≈ 2 * params[@vn(a.z0)]

        scoped = @inferred WrenCore.scope(GQAccumulator((:a,)), @vn(a.b[1]))
        @test scoped isa GQAccumulator{:all}
        scoped = @inferred WrenCore.scope(GQAccumulator((:a,)), @vn(b))
        @test scoped isa GQAccumulator{()}
    end

    @testset "submodel result used only in a generated quantity" begin
        @wren function gq_uses_submodel()
            x ~ gq_inner()
            @gq doubled = 2 * x[@vn(z0)]
        end
        params = rand(gq_uses_submodel())
        @test params[@vn(doubled)] ≈ 2 * params[@vn(x.z0)]
    end

    @testset "entry points" begin
        model = gq_demo([1.0, 2.0])
        @test haskey(rand(model), @vn(mu))

        accs = rand_with_logp(model)
        @test Set(keys(generated_quantities(accs))) ==
              Set([@vn(mu), @vn(mu1), @vn(s), @vn(r)])
        @test haskey(parameters(accs), @vn(a))
        @test !haskey(parameters(accs), @vn(mu))

        draw = Draw(FromPrior(), model)
        @test haskey(parameters(draw), @vn(mu))
        @test haskey(parameters(draw), @vn(a))

        # A generated quantity shares the output with the parameters; it wins a name clash.
        @wren function gq_clash()
            a ~ Normal()
            @gq a = 100.0
        end
        @test rand(gq_clash())[@vn(a)] == 100.0
    end

    @testset "VecModel does not evaluate generated quantities" begin
        vecmodel = VecModel(gq_throws(), nothing)
        @test lp(vecmodel, [0.1]) ≈ logpdf(Normal(), 0.1)
        @test logjoint(gq_throws(), VarNamedTuple(; a=0.1)) ≈ logpdf(Normal(), 0.1)
    end

    @testset "compile-time pruning" begin
        model = gq_pruned([1.0, 2.0])
        function typed_body(accs)
            argtypes = map(
                typeof,
                (
                    model.evaluation_fn,
                    FromPrior(),
                    OriginalSpace(),
                    accs,
                    model.args,
                    model.conditioned,
                    model.enzyme_copy_argnames,
                    EvalContext{false}(),
                ),
            )
            return only(
                code_typed(
                    WrenCore.RuntimeGeneratedFunctions.generated_callfunc,
                    Tuple{argtypes...},
                ),
            ).first
        end
        function calls_helper(code_info)
            return any(stmt -> occursin("gq_helper", string(stmt)), code_info.code)
        end

        @test !calls_helper(typed_body(AccumulatorTuple(ParamsAccumulator())))
        @test !calls_helper(typed_body(AccumulatorTuple(GQAccumulator((:other,)))))
        @test calls_helper(typed_body(AccumulatorTuple(GQAccumulator())))
        @test calls_helper(typed_body(AccumulatorTuple(GQAccumulator((:h,)))))
    end

    @testset "type stability" begin
        @inferred rand(gq_demo([1.0, 2.0]))
        @inferred rand(gq_outer())
        @inferred rand_with_logp(gq_demo([1.0, 2.0]))
    end
end

end # module WrenGQTests
