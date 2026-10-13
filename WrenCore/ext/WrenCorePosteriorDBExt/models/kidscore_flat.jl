# Four public Kidscore regressions with flat beta and positive flat sigma priors.
module KidscoreFlatCommon

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "kidiq_with_mom_work-kidscore_interaction_c",
    )))
end

@wren function kidscore_flat(kid_score, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    log_sigma ~ WD.Flat()
    sigma = exp(log_sigma)
    # A flat density in positive sigma induces this Jacobian in log sigma.
    @addloglikelihood log_sigma
    mu = X * beta
    @batch for i in eachindex(kid_score)
        @observe kid_score[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

end

module KidscoreInteractionC
using Statistics: mean
using ..KidscoreFlatCommon: data, kidscore_flat
const wren_model = let
    hs = data["mom_hs"] .- mean(data["mom_hs"])
    iq = data["mom_iq"] .- mean(data["mom_iq"])
    kidscore_flat(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreInteractionC2
using ..KidscoreFlatCommon: data, kidscore_flat
const wren_model = let
    hs = data["mom_hs"] .- 0.5
    iq = data["mom_iq"] .- 100.0
    kidscore_flat(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreInteractionZ
using Statistics: mean, std
using ..KidscoreFlatCommon: data, kidscore_flat
const wren_model = let
    hs = (data["mom_hs"] .- mean(data["mom_hs"])) ./ (2 * std(data["mom_hs"]))
    iq = (data["mom_iq"] .- mean(data["mom_iq"])) ./ (2 * std(data["mom_iq"]))
    kidscore_flat(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreMomWork
using ..KidscoreFlatCommon: data, kidscore_flat
const wren_model = let
    work = data["mom_work"]
    kidscore_flat(data["kid_score"], hcat(
        ones(length(work)), work .== 2, work .== 3, work .== 4,
    ))
end
end
