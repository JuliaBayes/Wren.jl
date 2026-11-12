module WrenSpaceTests

@info "testing space.jl"

using WrenCore
using Test
using Distributions

@testset "space.jl" begin
    ctx = EvalContext{false}()
    @testset "every latent representation agrees in every parameter space" begin
        @wren function exp_model()
            x ~ Exponential(1.0)
        end
        model = exp_model()

        # The unconstrained transform for Exponential is y = log(x), so the forward
        # log-Jacobian is log|dy/dx| = -y, and the density of y is exp(y) * p(exp(y)).
        x = 2.0
        y = log(x)
        representations = [
            TransformedValue(x, AsOriginal()),
            TransformedValue([x], AsVec()),
            TransformedValue([y], AsUnconstrainedVec()),
        ]
        @testset "$(typeof(tval.transform)) in $space" for tval in representations,
            space in (UnconstrainedSpace(), OriginalSpace())

            source = FromValues(VarNamedTuple(; x=tval))
            accs = AccumulatorTuple(
                LogPriorAccumulator(),
                LogJacobianAccumulator(),
                LogLikelihoodAccumulator(),
                ParamsAccumulator(),
            )
            _, accs = evaluate(model, source, space, accs, ctx)
            @test parameters(accs)[@vn(x)] ≈ x
            @test logprior(accs) ≈ -x
            if space isa UnconstrainedSpace
                @test logjac(accs) ≈ -y
                @test logjoint_transformed(accs) ≈ y - exp(y)
            else
                @test iszero(logjac(accs))
                @test logjoint_transformed(accs) ≈ -x
            end
        end
    end

    @testset "FromValues scoping and fallback in submodels" begin
        @wren function inner()
            x ~ Normal()
            y ~ Normal()
        end
        @wren function outer(sub)
            a ~ sub
        end
        model = outer(inner())

        partial = VarNamedTuple(; a=VarNamedTuple(; x=0.5))
        source = FromValues(partial, FromPrior())
        result = simulate(model, source)
        pars = parameters(result)
        @test pars[@vn(a.x)] == 0.5
        @test logprior(result) ≈ logpdf(Normal(), 0.5) + logpdf(Normal(), pars[@vn(a.y)])

        # No params at all under `a`: the submodel is evaluated with the fallback only.
        source = FromValues(VarNamedTuple(; b=1.0), FromPrior())
        result = simulate(model, source)
        @test haskey(parameters(result), @vn(a.x))
        @test haskey(parameters(result), @vn(a.y))

        @test_throws "No latent value provided for variable y" simulate(model, partial)
        @test_throws ArgumentError simulate(model, VarNamedTuple(; a=0.5))
    end
end

end # module WrenSpaceTests
