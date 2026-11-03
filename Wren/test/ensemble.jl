module WrenEnsembleTests

using Wren, Wren.Dists, EnsembleMCMC, FlexiChains, Random, Test

@wren function poisson_rate(count)
    rate ~ Exponential()
    @observe count ~ Poisson(rate)
end

@wren function buffered_normal(buffer)
    x ~ Normal()
    buffer[1] = x
    y = 1.0
    @observe y ~ Normal(buffer[1])
end

@testset "ensemble.jl" begin
    @testset "Transforms, axes and retention" begin
        model = poisson_rate(3)
        initial = [VarNamedTuple(rate=r + c) for r in (0.2, 0.7, 1.2, 2.7), c in (0.0, 0.3)]
        saved = deepcopy(initial)
        move = MoveMixture((StretchMove(), DEMove()), [2, 1]; schedule=:cycle)
        rng = Xoshiro(19)
        walkers = ensemble(
            rng,
            model,
            25,
            2;
            initial_params=initial,
            move,
            warmup=5,
            thin=2,
            progress=false,
        )
        reference_rng = Xoshiro(19)
        references = map(1:2) do chain
            vm = WrenCore.VecModel(model, nothing; latent_source=FromValues(initial[1, chain]))
            points = [
                WrenCore.vectorise_latents(FromValues(value), vm) for
                value in initial[:, chain]
            ]
            EnsembleMCMC.sample(
                reference_rng,
                x -> WrenCore.lp(vm, x),
                points,
                50;
                move,
                warmup=5,
            )
        end
        @test initial == saved
        @test rand(reference_rng, UInt64) == rand(rng, UInt64)
        for walker in 1:4
            chain = walkers[walker]
            rates = Array(chain[@vn(rate)])
            expected =
                hcat((exp.(ref.positions[1, walker, 2:2:50]) for ref in references)...)
            @test rates ≈ expected
            @test Array(chain[FlexiChains.Extra(:log_density)]) ≈
                  4 .* log.(rates) .- 2 .* rates .- log(6)
            @test Array(chain[FlexiChains.Extra(:logjoint)]) ≈
                  3 .* log.(rates) .- 2 .* rates .- log(6)
            @test Array(chain[FlexiChains.Extra(:accepted)]) ==
                  hcat((ref.accepted[walker, 2:2:50] for ref in references)...)
            @test Array(chain[FlexiChains.Extra(:move_index)]) ==
                  hcat((ref.move_indices[2:2:50] for ref in references)...)
            @test Array(chain[FlexiChains.Extra(:walker_id)]) == fill(walker, 25, 2)
            @test Array(chain[FlexiChains.Extra(:sweep)]) == repeat(collect(7:2:55), 1, 2)
        end
        single = ensemble(
            Xoshiro(19),
            model,
            25;
            initial_params=initial[:, 1],
            move,
            warmup=5,
            thin=2,
            progress=false,
        )
        @test all(
            Array(single[w][@vn(rate)])[:, 1] == Array(walkers[w][@vn(rate)])[:, 1] for
            w in 1:4
        )
    end

    @testset "Initialization RNG and model ownership" begin
        buffer = [100.0]
        model = buffered_normal(buffer)
        rng = Xoshiro(23)
        first_run = ensemble(rng, model, 20, 2; warmup=5, progress=false)
        replay = ensemble(Xoshiro(23), model, 20, 2; warmup=5, progress=false)
        continued = ensemble(rng, model, 20, 2; warmup=5, progress=false)
        @test all(FlexiChains.has_same_data.(first_run, replay))
        @test all(.!FlexiChains.has_same_data.(first_run, continued))
        @test Array(first_run[1][@vn(x)])[:, 1] != Array(first_run[1][@vn(x)])[:, 2]
        @test buffer == [100.0]
        @test size(Array(first(ensemble(model, 2; progress=false))[@vn(x)])) == (2, 1)
    end
end

end
