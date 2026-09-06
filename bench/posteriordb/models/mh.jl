# PosteriorDB: Mh_data-Mh_model
module Mh

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LinearAlgebra: I
using LogExpFunctions: log1pexp, logaddexp, logit
using Wren: Wren, @wren, @observe, @addloglikelihood
using ProbabilityMeasures: ProbabilityMeasures as PM

using ..PMProducts: ProductBinomialLogit, binomial_logconst

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Mh_data-Mh_model")))
end

# ── DynamicPPL ──

@model function pdb_Mh_model(M, T, y)
    @assert size(y) == (M,)
    omega ~ Uniform(0, 1)
    mean_p ~ Uniform(0, 1)
    sigma ~ Uniform(0, 5)
    eps_raw ~ product_distribution(Fill(Normal(0, 1), M))
    eps = logit(mean_p) .+ sigma .* eps_raw
    for i in 1:M
        if y[i] > 0
            DynamicPPL.@addlogprob! logpdf(Bernoulli(omega), 1) +
                                    logpdf(BinomialLogit(T, eps[i]), y[i])
        else
            DynamicPPL.@addlogprob! logaddexp(
                logpdf(Bernoulli(omega), 1) + logpdf(BinomialLogit(T, eps[i]), 0),
                logpdf(Bernoulli(omega), 0),
            )
        end
    end
end

const turing_model = pdb_Mh_model(data["M"], data["T"], data["y"])

# ── Wren ──

@wren function mh_model(captures, detected, undetected, trials, prior_mean, logconst)
    omega ~ PM.Uniform(0.0, 1.0)
    mean_p ~ PM.Uniform(0.0, 1.0)
    sigma ~ PM.Uniform(0.0, 5.0)
    eps_raw ~ PM.MvNormal(prior_mean, 1.0 * I)
    logit_p = logit(mean_p) .+ sigma .* eps_raw
    @observe captures ~ ProductBinomialLogit(trials, logit_p[detected], logconst)
    @addloglikelihood length(detected) * log(omega) + sum(
        logaddexp.(log(omega) .- trials .* log1pexp.(logit_p[undetected]), log1p(-omega)),
    )
    return nothing
end

const wren_model = let
    M = data["M"]
    trials = data["T"]
    y = data["y"]
    detected = findall(>(0), y)
    undetected = findall(==(0), y)
    captures = y[detected]
    mh_model(
        captures,
        detected,
        undetected,
        trials,
        zeros(M),
        binomial_logconst(trials, captures),
    )
end

end
