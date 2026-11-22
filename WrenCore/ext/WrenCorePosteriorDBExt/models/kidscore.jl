# PosteriorDB: the Kidscore regressions. There are two groups:
#
#  - kidiq-kidscore_*: four regressions on the `kidiq` dataset, with flat beta and a
#    half-Cauchy prior on sigma.
#  - kidiq_with_mom_work-kidscore_*: four regressions on the `kidiq_with_mom_work` dataset,
#    with flat beta and sigma priors.
#
# Within each group, the models differ only in the predictors.
module Kidscore

using PosteriorDB
using Statistics: mean, std
using WrenCore: @wren
using WrenDists: WrenDists as WD

load_data(name) = PosteriorDB.load(
    PosteriorDB.dataset(PosteriorDB.posterior(PosteriorDB.database(), name)),
)

@wren function kidiq_base(kid_score, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    sigma ~ WD.HalfCauchy(0.0, 2.5)
    mu = X * beta
    @batch for i in eachindex(kid_score)
        @observe kid_score[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

@wren function kidiq_with_mom_work_base(kid_score, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    sigma ~ WD.PositiveFlat()
    mu = X * beta
    @batch for i in eachindex(kid_score)
        @observe kid_score[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const kidiq = load_data("kidiq-kidscore_interaction")
const kidiq_with_mom_work = load_data("kidiq_with_mom_work-kidscore_interaction_c")

# kidiq-kidscore_interaction
const interaction = let
    hs, iq = kidiq["mom_hs"], kidiq["mom_iq"]
    kidiq_base(kidiq["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
# kidiq-kidscore_momhs
const momhs = let
    hs = kidiq["mom_hs"]
    kidiq_base(kidiq["kid_score"], hcat(ones(length(hs)), hs))
end
# kidiq-kidscore_momiq
const momiq = let
    iq = kidiq["mom_iq"]
    kidiq_base(kidiq["kid_score"], hcat(ones(length(iq)), iq))
end
# kidiq-kidscore_momhsiq
const momhsiq = let
    hs, iq = kidiq["mom_hs"], kidiq["mom_iq"]
    kidiq_base(kidiq["kid_score"], hcat(ones(length(hs)), hs, iq))
end

# kidiq_with_mom_work-kidscore_interaction_c
const interaction_c = let
    d = kidiq_with_mom_work
    hs = d["mom_hs"] .- mean(d["mom_hs"])
    iq = d["mom_iq"] .- mean(d["mom_iq"])
    kidiq_with_mom_work_base(d["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
# kidiq_with_mom_work-kidscore_interaction_c2
const interaction_c2 = let
    d = kidiq_with_mom_work
    hs = d["mom_hs"] .- 0.5
    iq = d["mom_iq"] .- 100.0
    kidiq_with_mom_work_base(d["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
# kidiq_with_mom_work-kidscore_interaction_z
const interaction_z = let
    d = kidiq_with_mom_work
    hs = (d["mom_hs"] .- mean(d["mom_hs"])) ./ (2 * std(d["mom_hs"]))
    iq = (d["mom_iq"] .- mean(d["mom_iq"])) ./ (2 * std(d["mom_iq"]))
    kidiq_with_mom_work_base(d["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
# kidiq_with_mom_work-kidscore_mom_work
const mom_work = let
    d = kidiq_with_mom_work
    work = d["mom_work"]
    kidiq_with_mom_work_base(
        d["kid_score"],
        hcat(ones(length(work)), work .== 2, work .== 3, work .== 4),
    )
end

end
