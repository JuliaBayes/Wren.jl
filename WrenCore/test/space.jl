module WrenSpaceTests

@info "testing space.jl"

using WrenCore
using Test
using Distributions

@testset "space.jl" begin
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

            source = FromParams(VarNamedTuple(; x=tval))
            _, accs = evaluate(model, source, space, WrenCore.default_accumulators())
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

    @testset "FromParams scoping and fallback in submodels" begin
        @wren function inner()
            x ~ Normal()
            y ~ Normal()
        end
        @wren function outer(sub)
            a ~ sub
        end
        model = outer(inner())
        accs = WrenCore.default_accumulators()

        partial = VarNamedTuple(; a=VarNamedTuple(; x=0.5))
        source = FromParams(partial, FromPrior())
        _, new_accs = evaluate(model, source, OriginalSpace(), accs)
        pars = parameters(new_accs)
        @test pars[@vn(a.x)] == 0.5
        @test logprior(new_accs) ≈ logpdf(Normal(), 0.5) + logpdf(Normal(), pars[@vn(a.y)])

        # No params at all under `a`: the submodel is evaluated with the fallback only.
        source = FromParams(VarNamedTuple(; b=1.0), FromPrior())
        _, new_accs = evaluate(model, source, OriginalSpace(), accs)
        @test haskey(parameters(new_accs), @vn(a.x))
        @test haskey(parameters(new_accs), @vn(a.y))

        @test_throws "No value was provided for variable y" evaluate(
            model,
            FromParams(partial),
            OriginalSpace(),
            accs,
        )
        @test_throws ArgumentError evaluate(
            model,
            FromParams(VarNamedTuple(; a=0.5)),
            OriginalSpace(),
            accs,
        )
    end
end

end # module WrenSpaceTests
