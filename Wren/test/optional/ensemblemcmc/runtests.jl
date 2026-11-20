module WrenEnsembleTests

using Wren, Wren.Dists, EnsembleMCMC, FlexiChains, Random, Test
using ADTypes: AutoReactant
using Reactant: Reactant

get(ENV, "WREN_TEST_BACKEND", "cpu") == "gpu" && (@eval using CUDA)
Reactant.set_default_backend(get(ENV, "WREN_TEST_BACKEND", "cpu"))

@wren function poisson_rate(count, buffer)
    rate ~ Exponential()
    buffer .= rate
    @observe count ~ Poisson(sum(buffer))
end

@testset "ensemble.jl" begin
    buffer = [-1.0]
    model = poisson_rate(3, buffer)
    options = (; warmup=3, thin=2, progress=false)
    walkers = ensemble(Xoshiro(19), model, 8, 2; options...)
    replay = ensemble(Xoshiro(19), model, 8, 2; options...)
    @test buffer == [-1.0]
    @test length(walkers) == 4
    @test all(FlexiChains.has_same_data.(walkers, replay))
    for chain in walkers
        rates = Array(chain[@vn(rate)])
        @test size(rates) == (8, 2)
        @test rates[:, 1] != rates[:, 2]
        @test Array(chain[FlexiChains.Extra(:log_density)]) ≈
              4 .* log.(rates) .- 2 .* rates .- log(6)
        @test Array(chain[FlexiChains.Extra(:logjoint)]) ≈
              3 .* log.(rates) .- 2 .* rates .- log(6)
        @test Array(chain[FlexiChains.Extra(:sweep)]) == repeat(collect(5:2:19), 1, 2)
    end

    rates = [0.2 0.3; 0.7 0.8; 1.2 1.3; 2.7 2.8]
    initial = [VarNamedTuple(rate=r) for r in rates]
    # Near-identity proposals expose the supplied starting values.
    move = StretchMove(scale=nextfloat(1.0))
    explicit = ensemble(model, 1, 2; initial_params=initial, move, progress=false)
    @test reduce(vcat, (Array(w[@vn(rate)]) for w in explicit)) ≈ rates
    single = ensemble(model, 1; initial_params=initial[:, 1], move, progress=false)
    @test reduce(vcat, (Array(w[@vn(rate)]) for w in single)) ≈ rates[:, 1:1]

    adtype = AutoReactant(AutoEnzyme(; mode=Reverse))
    compiled_initial =
        ensemble(model, 1, 2; adtype, initial_params=initial, move, progress=false)
    @test reduce(vcat, (Array(w[@vn(rate)]) for w in compiled_initial)) ≈ rates
    compiled = ensemble(Xoshiro(19), model, 2, 2; adtype, progress=false)
    for chain in compiled
        values = Array(chain[@vn(rate)])
        @test Array(chain[FlexiChains.Extra(:log_density)]) ≈
              4 .* log.(values) .- 2 .* values .- log(6)
    end
    @test buffer == [-1.0]
end

end
