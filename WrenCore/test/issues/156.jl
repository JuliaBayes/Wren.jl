module Issue156

using WrenCore, WrenDists, Enzyme, Test

@testset "#156 Evaluating a model doesn't write into the caller's latents" begin
    # In both models the unconstrained transform of `x` is the identity, so without a copy
    # the model's `x` is a view of the parameter vector.
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

    @testset "$(nameof(f))" for f in (mutate_mvnormal, mutate_iid)
        model = f()
        initial = VarNamedTuple(; x=[2.0, 3.0])
        # x = [3, 3] after the update; the observe contributes logpdf(Normal(3, 1), 0).
        expected = logjoint(model, initial)
        @test initial[@vn(x)] == [2.0, 3.0]

        @testset "adtype=$adtype" for adtype in (nothing, WrenCore.DEFAULT_AD)
            vm = VecModel(model, adtype; latent_source=FromParams(initial))
            @test initial[@vn(x)] == [2.0, 3.0]
            p = [2.0, 3.0]
            @test lp(vm, p) ≈ expected
            @test lp(vm, p) ≈ expected
            @test p == [2.0, 3.0]
            if adtype !== nothing
                @test first(lp_and_grad(vm, p)) ≈ expected
                @test p == [2.0, 3.0]
            end
            Draw(p, vm)
            @test p == [2.0, 3.0]
        end
    end
end

end
