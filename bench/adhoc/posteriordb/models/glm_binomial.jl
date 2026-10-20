# PosteriorDB: GLM_Binomial_data-GLM_Binomial_model
module GLMBinomial

using PosteriorDB
using DynamicPPL, Distributions
using Wren: Wren, @wren, @observe, @batch
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLM_Binomial_data-GLM_Binomial_model"),
        ),
    )
end

# ── DynamicPPL ──

@model function pdb_GLM_Binomial_model(nyears, C, N, year, year_squared)
    alpha ~ Normal(0, 100)
    beta1 ~ Normal(0, 100)
    beta2 ~ Normal(0, 100)
    logit_p = alpha .+ beta1 .* year .+ beta2 .* year_squared
    DynamicPPL.@addlogprob! logpdf(product_distribution(map(BinomialLogit, N, logit_p)), C)
end

const turing_model = let
    year = data["year"]
    pdb_GLM_Binomial_model(data["nyears"], data["C"], data["N"], year, year .^ 2)
end

# ── Wren ──

@wren function glm_binomial_model(counts, trials, year, year_squared)
    alpha ~ WD.Normal(0.0, 100.0)
    beta1 ~ WD.Normal(0.0, 100.0)
    beta2 ~ WD.Normal(0.0, 100.0)
    logit_p = alpha .+ beta1 .* year .+ beta2 .* year_squared
    @batch for i in eachindex(counts)
        @observe counts[i] ~ WD.BinomialLogit(trials[i], logit_p[i])
    end
    return nothing
end

const wren_model = let
    counts = data["C"]
    trials = data["N"]
    year = data["year"]
    glm_binomial_model(counts, trials, year, year .^ 2)
end

end
