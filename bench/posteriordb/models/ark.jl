# PosteriorDB: arK-arK
# Deviations: sigma prior is half-Cauchy(0, 2.5) in PDB, replaced with Exponential(2.5)
# as PM doesn't have truncated distributions. Applied to both models for fairness.
module ArK

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LinearAlgebra: I
using Wren: Wren, @wren, @observe
using ProbabilityMeasures: ProbabilityMeasures as PM

using ..PMProducts: ProductNormal

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "arK-arK")))
end

# ── DynamicPPL ──

@model function arK_arK(y, lags, K)
    alpha ~ Normal(0, 10)
    beta ~ product_distribution(Fill(Normal(0, 10), K))
    sigma ~ Exponential(2.5)
    mu = alpha .+ lags * beta
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), length(y))),
        y .- mu,
    )
end

const turing_model = let
    K = data["K"]
    T = data["T"]
    series = data["y"]
    lags = [series[t-k] for t in (K+1):T, k in 1:K]
    arK_arK(series[(K+1):T], lags, K)
end

# ── Wren ──

@wren function ark_model(y, lags, prior_mean)
    alpha ~ PM.Normal(0.0, 10.0)
    beta ~ PM.MvNormal(prior_mean, 10.0 * I)
    sigma ~ PM.Exponential(2.5)
    @observe y ~ ProductNormal(alpha .+ lags * beta, sigma)
    return nothing
end

const wren_model = let
    K = data["K"]
    T = data["T"]
    series = data["y"]
    # Row t of `lags` holds the K values preceding observation t.
    lags = [series[t-k] for t in (K+1):T, k in 1:K]
    ark_model(series[(K+1):T], lags, zeros(K))
end

end
