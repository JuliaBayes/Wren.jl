# PosteriorDB: GLMM_Poisson_data-GLMM_Poisson_model
# Deviations: eps prior is Flat() in PDB, replaced with MvNormal(0, 100I)
# as WrenDists doesn't have improper priors. Applied to both models for fairness.
module GLMMPoisson

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LinearAlgebra: I
using Wren: Wren, @wren, @observe, @batch, @iid, @addloglikelihood, logdensityof
using WrenCore: iid
using WrenDists: WrenDists as WD

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
    eps ~ MvNormal(zeros(n), 100 * I)
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

@wren function glmm_poisson_model(counts, year, year_squared, year_cubed, n)
    alpha ~ WD.Uniform(-20.0, 20.0)
    beta1 ~ WD.Uniform(-10.0, 10.0)
    beta2 ~ WD.Uniform(-10.0, 20.0)
    beta3 ~ WD.Uniform(-10.0, 10.0)
    sigma ~ WD.Uniform(0.0, 5.0)
    @iid eps ~ WD.Normal(0.0, 10.0) size=n
    log_lambda =
        alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed .+ eps
    @batch for i in eachindex(counts)
        @observe counts[i] ~ WD.PoissonLog(log_lambda[i])
    end
    @addloglikelihood logdensityof(iid(WD.Normal(0.0, sigma), size(eps)), eps)
    return nothing
end

const wren_model = let
    counts = data["C"]
    year = data["year"]
    glmm_poisson_model(counts, year, year .^ 2, year .^ 3, data["n"])
end

end
