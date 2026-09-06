# PosteriorDB: GLMM_Poisson_data-GLMM_Poisson_model
module GLMMPoisson

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LinearAlgebra: I
using Wren: Wren, @wren, @observe, @addloglikelihood, logdensityof
using ProbabilityMeasures: ProbabilityMeasures as PM

using ..PMProducts: ProductPoisson, poisson_logconst

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLMM_Poisson_data-GLMM_Poisson_model"),
        ),
    )
end

# ── DynamicPPL ──

@model function pdb_GLMM_Poisson_model(n, C, year, year_squared, year_cubed)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 20)
    beta3 ~ Uniform(-10, 10)
    eps ~ product_distribution(Fill(Flat(), n))
    sigma ~ Uniform(0, 5)
    log_lambda =
        alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed .+ eps
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(x -> Poisson(exp(x)), log_lambda)),
        C,
    )
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(eps))),
        eps,
    )
end

const turing_model = let
    year = data["year"]
    pdb_GLMM_Poisson_model(data["n"], data["C"], year, year .^ 2, year .^ 3)
end

# ── Wren ──

@wren function glmm_poisson_model(
    counts,
    year,
    year_squared,
    year_cubed,
    prior_mean,
    logconst,
)
    alpha ~ PM.Uniform(-20.0, 20.0)
    beta1 ~ PM.Uniform(-10.0, 10.0)
    beta2 ~ PM.Uniform(-10.0, 20.0)
    beta3 ~ PM.Uniform(-10.0, 10.0)
    sigma ~ PM.Uniform(0.0, 5.0)
    eps ~ PM.MvNormal(prior_mean, 100.0 * I) # substitutes Flat()
    log_lambda =
        alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed .+ eps
    @observe counts ~ ProductPoisson(exp.(log_lambda), logconst)
    @addloglikelihood logdensityof(PM.MvNormal(prior_mean, sigma * I), eps)
    return nothing
end

const wren_model = let
    counts = data["C"]
    year = data["year"]
    glmm_poisson_model(
        counts,
        year,
        year .^ 2,
        year .^ 3,
        zeros(data["n"]),
        poisson_logconst(counts),
    )
end

end
