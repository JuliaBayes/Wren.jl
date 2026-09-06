# PosteriorDB: GLM_Binomial_data-GLM_Binomial_model
module GLMBinomial

using PosteriorDB
using DynamicPPL, Distributions
using Wren: Wren, @wren, @observe
using ProbabilityMeasures: ProbabilityMeasures as PM

using ..PMProducts: ProductBinomialLogit, binomial_logconst

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

@wren function glm_binomial_model(counts, trials, year, year_squared, logconst)
    alpha ~ PM.Normal(0.0, 100.0)
    beta1 ~ PM.Normal(0.0, 100.0)
    beta2 ~ PM.Normal(0.0, 100.0)
    logit_p = alpha .+ beta1 .* year .+ beta2 .* year_squared
    @observe counts ~ ProductBinomialLogit(trials, logit_p, logconst)
    return nothing
end

const wren_model = let
    counts = data["C"]
    trials = data["N"]
    year = data["year"]
    glm_binomial_model(counts, trials, year, year .^ 2, binomial_logconst(trials, counts))
end

end
