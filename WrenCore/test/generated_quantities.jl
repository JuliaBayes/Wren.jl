module WrenCoreGeneratedQuantitiesTests

@info "testing generated_quantities.jl"

using AbstractMCMC: AbstractMCMC
using Distributions
using FlexiChains: FlexiChains, Parameter, VNChain
using StableRNGs
using Test
using WrenCore

@wren function gq_sub()
    z0 ~ Normal()
    @gq z = 2z0
end

@wren function gq_model()
    x ~ Normal()
    a ~ gq_sub()
    @gq x2 = x^2
    @gq xp1 = x + 1
    y ~ Normal(x)
end

@wren function sub_without_gqs()
    z0 ~ Normal()
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

function prior_chain(rng, model, niters, nchains)
    draws = map(Iterators.product(1:niters, 1:nchains)) do _
        Draw(FromPrior(rng), model)
    end
    return AbstractMCMC.from_samples(VNChain, draws)
end

parameter_keys(chain) = Set(FlexiChains.parameters(chain))

@testset "generated_quantities" begin
    model = gq_model()
    chn = prior_chain(StableRNG(468), gq_model_without_gqs(), 10, 2)

    gqs_all = generated_quantities(model, chn)
    gqs_some = generated_quantities(model, chn; to_collect=Some([:x2, :a]))
    gqs_vec = generated_quantities(model, chn; to_collect=[:x2])

    @testset "keys" begin
        @test parameter_keys(gqs_all) == Set([@vn(x2), @vn(xp1), @vn(a.z)])
        @test parameter_keys(gqs_some) == Set([@vn(x2), @vn(a.z)])
        @test parameter_keys(gqs_vec) == Set([@vn(x2)])
        @test generated_quantities(model, chn; to_collect=All()) isa VNChain
    end

    @testset "forms agree on shared keys" begin
        for vn in parameter_keys(gqs_some)
            @test gqs_some[vn] == gqs_all[vn]
        end
        @test gqs_vec[@vn(x2)] == gqs_all[@vn(x2)]
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

    @testset "invalid to_collect" begin
        msg = "`to_collect` must be `All()`, `Some(names)` or `names`"
        @test_throws msg generated_quantities(model, chn; to_collect=:x2)
        @test_throws msg generated_quantities(model, chn; to_collect=["x2"])
        @test_throws ArgumentError generated_quantities(model, chn; to_collect=:x2)
    end

    @testset "missing parameter errors" begin
        chn_without_a = prior_chain(StableRNG(468), gq_model_without_submodel(), 10, 2)
        @test_throws "No latent value provided for variable z0" generated_quantities(
            model,
            chn_without_a,
        )
    end

    @testset "predict holds generated quantities" begin
        pdns = predict(StableRNG(468), model, chn)
        @test issubset(Set([@vn(x2), @vn(xp1), @vn(a.z)]), parameter_keys(pdns))
        @test pdns[@vn(x2)] == chn[@vn(x)] .^ 2
    end
end

end # module WrenCoreGeneratedQuantitiesTests
