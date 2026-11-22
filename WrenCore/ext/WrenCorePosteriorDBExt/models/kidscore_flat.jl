# PosteriorDB: four kidiq_with_mom_work regressions with flat beta and sigma priors. They
# share a dataset and differ only in the predictors, so each module just builds its own `X`.
# (Unlike `kidscore_interaction`, these have no prior on sigma.)
module KidscoreWorkBase

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "kidiq_with_mom_work-kidscore_interaction_c"),
        ),
    )
end

@wren function kidscore_work_base(kid_score, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    sigma ~ WD.PositiveFlat()
    mu = X * beta
    @batch for i in eachindex(kid_score)
        @observe kid_score[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

end

module KidscoreInteractionC
using Statistics: mean
using ..KidscoreWorkBase: data, kidscore_work_base
const wren_model = let
    hs = data["mom_hs"] .- mean(data["mom_hs"])
    iq = data["mom_iq"] .- mean(data["mom_iq"])
    kidscore_work_base(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreInteractionC2
using ..KidscoreWorkBase: data, kidscore_work_base
const wren_model = let
    hs = data["mom_hs"] .- 0.5
    iq = data["mom_iq"] .- 100.0
    kidscore_work_base(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreInteractionZ
using Statistics: mean, std
using ..KidscoreWorkBase: data, kidscore_work_base
const wren_model = let
    hs = (data["mom_hs"] .- mean(data["mom_hs"])) ./ (2 * std(data["mom_hs"]))
    iq = (data["mom_iq"] .- mean(data["mom_iq"])) ./ (2 * std(data["mom_iq"]))
    kidscore_work_base(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreMomWork
using ..KidscoreWorkBase: data, kidscore_work_base
const wren_model = let
    work = data["mom_work"]
    kidscore_work_base(
        data["kid_score"],
        hcat(ones(length(work)), work .== 2, work .== 3, work .== 4),
    )
end
end
