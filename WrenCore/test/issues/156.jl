module Issue156

using WrenCore, WrenDists, Enzyme, Test
using AbstractMCMC: AbstractMCMC
using FlexiChains: VNChain
import LogDensityProblems as LDP

# Model evaluation doesn't copy latents, so the model is allowed to write into what it is
# given (e.g. the vector passed to `lp`). The LogDensityProblems methods, `Draw`,
# conditioning and the chain methods copy the caller's data once, so the model can't write
# into that.
@testset "#156 Evaluating a model doesn't write into the caller's latents" begin
    # In both models the unconstrained transform of `x` is the identity, so the model's `x`
    # is a view of the parameter vector.
    @wren function mutate_mvnormal()
        x ~ MvNormalScale(zeros(2), I)
        x[1] += 1.0
        y = 0.0
        @observe y ~ Normal(x[1], 1.0)
    end
    @wren function mutate_iid()
        x ~ iid(Normal(), (2,))
        x[1] += 1.0
        y = 0.0
        @observe y ~ Normal(x[1], 1.0)
    end
    # Writes into `x` without any syntax that looks like a write.
    @wren function mutate_foreach()
        x ~ MvNormalScale(zeros(2), I)
        foreach(fill!, (x,), (0.0,))
        y = 0.0
        @observe y ~ Normal(x[1], 1.0)
    end

    @testset "$(nameof(f))" for f in (mutate_mvnormal, mutate_iid, mutate_foreach)
        model = f()
        # A fresh value each time, since `logjoint(model, values)` may write into `values`.
        initial() = VarNamedTuple(; x=[2.0, 3.0])
        expected = logjoint(model, initial())

        @testset "adtype=$adtype" for adtype in (nothing, WrenCore.DEFAULT_AD)
            vm = VecModel(model, adtype; latent_source=FromValues(initial()))
            p = [2.0, 3.0]
            @test LDP.logdensity(vm, p) ≈ expected
            @test LDP.logdensity(vm, p) ≈ expected
            @test p == [2.0, 3.0]
            if adtype !== nothing
                @test first(LDP.logdensity_and_gradient(vm, p)) ≈ expected
                @test p == [2.0, 3.0]
            end
            Draw(p, vm)
            @test p == [2.0, 3.0]
            # `lp` itself makes no copy, but still gives the right answer for a fresh vector.
            @test lp(vm, [2.0, 3.0]) ≈ expected
        end

        @testset "conditioned values" begin
            x = [2.0, 3.0]
            conditioned = model | VarNamedTuple(; x=x)
            logjoint(conditioned, VarNamedTuple())
            @test x == [2.0, 3.0]
        end

        @testset "chain values" begin
            draws = [Draw(FromValues(initial()), model) for _ in 1:2, _ in 1:1]
            chn = AbstractMCMC.from_samples(VNChain, draws)
            before = deepcopy(chn[@vn(x)])
            predict(model, chn)
            predict(model, chn)
            @test chn[@vn(x)] == before
        end
    end
end

end
