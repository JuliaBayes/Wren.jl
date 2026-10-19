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
            vm = VecModel(model, adtype; latent_source=FromValues(initial))
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

        @testset "conditioned values" begin
            # The conditioned value is the caller's own `Array`, and the model writes into
            # it if it isn't copied.
            x = [2.0, 3.0]
            conditioned = model | VarNamedTuple(; x=x)
            rand(conditioned)
            rand(conditioned)
            @test x == [2.0, 3.0]
            @test conditioned.conditioned[@vn(x)] == [2.0, 3.0]
        end
    end

    @testset "mutation through a function bound in the model" begin
        @wren function mutate_closure()
            x ~ MvNormalScale(zeros(2), I)
            f = v -> (v .+= 1.0)
            f(x)
            y = 0.0
            @observe y ~ Normal(x[1], 1.0)
        end
        vm = VecModel(mutate_closure(), nothing)
        p = [2.0, 3.0]
        @test lp(vm, p) ≈ lp(vm, p)
        @test p == [2.0, 3.0]
    end

    @testset "read-only latents" begin
        @wren function read_only()
            x ~ MvNormalScale(zeros(2), I)
            y = 0.0
            @observe y ~ Normal(x[1], 1.0)
        end
        p = [2.0, 3.0]
        @test lp(VecModel(read_only(), nothing), p) ≈
              logjoint(read_only(), VarNamedTuple(; x=p))
        @test p == [2.0, 3.0]
    end

    @testset "parameters are recorded as sampled, before the model writes to them" begin
        vm = VecModel(mutate_mvnormal(), nothing)
        @test parameters(Draw([2.0, 3.0], vm))[@vn(x)] == [2.0, 3.0]
    end
end

end
