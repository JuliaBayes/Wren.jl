# PosteriorDB: kidiq-kidscore_momhs, kidiq-kidscore_momiq and kidiq-kidscore_momhsiq.
# These are the same model as kidiq-kidscore_interaction, on the same dataset, with
# different predictors.
module KidscoreMom

using ..KidscoreInteraction: data, kidscore_interaction

const kid_score = data["kid_score"]
const mom_hs = data["mom_hs"]
const mom_iq = data["mom_iq"]
const intercept = ones(length(kid_score))

# kidiq-kidscore_momhs
const momhs = kidscore_interaction(kid_score, hcat(intercept, mom_hs))
# kidiq-kidscore_momiq
const momiq = kidscore_interaction(kid_score, hcat(intercept, mom_iq))
# kidiq-kidscore_momhsiq
const momhsiq = kidscore_interaction(kid_score, hcat(intercept, mom_hs, mom_iq))

end
