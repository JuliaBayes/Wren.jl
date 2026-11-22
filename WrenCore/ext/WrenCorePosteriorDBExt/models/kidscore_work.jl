# PosteriorDB: the four kidiq_with_mom_work regressions, with flat beta and sigma priors.
# They share a dataset and differ only in the predictors. (Unlike `kidscore_interaction`,
# these have no prior on sigma.)
module KidscoreWork

using PosteriorDB
using Statistics: mean, std
using WrenCore: @wren
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

const kid_score = data["kid_score"]
const mom_hs = data["mom_hs"]
const mom_iq = data["mom_iq"]
const mom_work = data["mom_work"]
const intercept = ones(length(kid_score))

# kidiq_with_mom_work-kidscore_interaction_c
const interaction_c = let
    hs = mom_hs .- mean(mom_hs)
    iq = mom_iq .- mean(mom_iq)
    kidscore_work_base(kid_score, hcat(intercept, hs, iq, hs .* iq))
end
# kidiq_with_mom_work-kidscore_interaction_c2
const interaction_c2 = let
    hs = mom_hs .- 0.5
    iq = mom_iq .- 100.0
    kidscore_work_base(kid_score, hcat(intercept, hs, iq, hs .* iq))
end
# kidiq_with_mom_work-kidscore_interaction_z
const interaction_z = let
    hs = (mom_hs .- mean(mom_hs)) ./ (2 * std(mom_hs))
    iq = (mom_iq .- mean(mom_iq)) ./ (2 * std(mom_iq))
    kidscore_work_base(kid_score, hcat(intercept, hs, iq, hs .* iq))
end
# kidiq_with_mom_work-kidscore_mom_work
const mom_work_model = kidscore_work_base(
    kid_score,
    hcat(intercept, mom_work .== 2, mom_work .== 3, mom_work .== 4),
)

end
