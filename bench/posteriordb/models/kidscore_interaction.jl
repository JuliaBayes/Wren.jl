# PosteriorDB: kidiq-kidscore_interaction
module KidscoreInteraction

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LinearAlgebra: I
using Wren: Wren, @wren, @observe
using ProbabilityMeasures: ProbabilityMeasures as PM

using ..PMProducts: ProductNormal

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kidiq-kidscore_interaction")),
    )
end

# ── DynamicPPL ──

@model function kidiq_kidscore_interaction(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 4))
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

@wren function kidscore_interaction(kid_score, X, prior_mean)
    beta ~ PM.MvNormal(prior_mean, 100.0 * I) # substitutes Flat()
    sigma ~ PM.Exponential(2.5) # substitutes truncated(Cauchy(0, 2.5); lower=0)
    @observe kid_score ~ ProductNormal(X * beta, sigma)
    return nothing
end

const wren_model = let
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    X = hcat(ones(length(mom_hs)), mom_hs, mom_iq, mom_hs .* mom_iq)
    kidscore_interaction(data["kid_score"], X, zeros(size(X, 2)))
end

end
