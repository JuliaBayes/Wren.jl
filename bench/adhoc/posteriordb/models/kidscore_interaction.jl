# PosteriorDB: kidiq-kidscore_interaction
# Deviations: beta prior is Flat() in PDB, replaced with MvNormal(0, 100I) as WrenDists
# doesn't have improper priors. Applied to both models for fairness.
module KidscoreInteraction

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LinearAlgebra: I
using Wren: Wren, @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kidiq-kidscore_interaction")),
    )
end

# ── DynamicPPL ──

@model function kidiq_kidscore_interaction(kid_score, X)
    beta ~ MvNormal(zeros(4), 100 * I)
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

const turing_model = let
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    X = hcat(ones(length(mom_hs)), mom_hs, mom_iq, mom_hs .* mom_iq)
    kidiq_kidscore_interaction(data["kid_score"], X)
end

# ── Wren ──

@wren function kidscore_interaction(kid_score, X)
    @iid beta ~ WD.Normal(0.0, 10.0) size=size(X, 2)
    sigma ~ WD.HalfCauchy(0.0, 2.5)
    mu = X * beta
    @batch for i in eachindex(kid_score)
        @observe kid_score[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const wren_model = let
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    X = hcat(ones(length(mom_hs)), mom_hs, mom_iq, mom_hs .* mom_iq)
    kidscore_interaction(data["kid_score"], X)
end

end
