# PosteriorDB: Mh_data-Mh_model
# Deviations: detected/undetected partitioning is done at model construction time
# rather than inside the model loop. Applied to both models for fairness.
module Mh

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LogExpFunctions: log1pexp, logaddexp, logit
using Wren: Wren, @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Mh_data-Mh_model")))
end

# ── DynamicPPL ──

@model function pdb_Mh_model(captures, detected, undetected, M, T)
    omega ~ Uniform(0, 1)
    mean_p ~ Uniform(0, 1)
    sigma ~ Uniform(0, 5)
    eps_raw ~ product_distribution(Fill(Normal(0, 1), M))
    logit_p = logit(mean_p) .+ sigma .* eps_raw
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(lp -> BinomialLogit(T, lp), logit_p[detected])),
        captures,
    )
    DynamicPPL.@addlogprob! length(detected) * log(omega) + sum(
        logaddexp.(log(omega) .- T .* log1pexp.(logit_p[undetected]), log1p(-omega)),
    )
end

const turing_model = let
    y = data["y"]
    detected = findall(>(0), y)
    undetected = findall(==(0), y)
    pdb_Mh_model(y[detected], detected, undetected, data["M"], data["T"])
end

# ── Wren ──

@wren function mh_model(captures, detected, undetected, trials, M)
    omega ~ WD.Uniform(0.0, 1.0)
    mean_p ~ WD.Uniform(0.0, 1.0)
    sigma ~ WD.Uniform(0.0, 5.0)
    @iid eps_raw ~ WD.Normal(0.0, 1.0) size=M
    logit_p = logit(mean_p) .+ sigma .* eps_raw
    logit_p_detected = logit_p[detected]
    @batch for i in eachindex(captures)
        @observe captures[i] ~ WD.BinomialLogit(trials, logit_p_detected[i])
    end
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
    mh_model(captures, detected, undetected, trials, M)
end

end
