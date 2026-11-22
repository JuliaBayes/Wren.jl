# PosteriorDB: the nine wells_data logistic regressions. They share a dataset and differ
# only in the predictors.
module Wells

using PosteriorDB
using Statistics: mean
using WrenCore: @wren
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "wells_data-wells_dist")),
    )
end

# wells_dist has no separate intercept: it's the first column of `X`.
@wren function wells_dist_model(switched, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    logit_p = X * beta
    @batch for i in eachindex(switched)
        @observe switched[i] ~ WD.BernoulliLogit(logit_p[i])
    end
    return nothing
end

@wren function wells_glm_model(switched, X)
    alpha ~ WD.Flat()
    @iid beta ~ WD.Flat() size=size(X, 2)
    logit_p = alpha .+ X * beta
    @batch for i in eachindex(switched)
        @observe switched[i] ~ WD.BernoulliLogit(logit_p[i])
    end
    return nothing
end

const switched = data["switched"]
const dist = data["dist"]
const arsenic = data["arsenic"]
const assoc = data["assoc"]
const educ = data["educ"]
const dist100 = dist ./ 100.0
const educ4 = educ ./ 4.0
const c_dist100 = (dist .- mean(dist)) ./ 100.0
const c_arsenic = arsenic .- mean(arsenic)
const c_educ4 = (educ .- mean(educ)) ./ 4.0

# wells_data-wells_dist
const wells_dist = wells_dist_model(switched, hcat(ones(length(dist)), dist))
# wells_data-wells_dist100_model
const dist100_model = wells_glm_model(switched, hcat(dist100))
# wells_data-wells_dist100ars_model
const dist100ars_model = wells_glm_model(switched, hcat(dist100, arsenic))
# wells_data-wells_dae_model
const dae_model = wells_glm_model(switched, hcat(dist100, arsenic, educ4))
# wells_data-wells_dae_c_model
const dae_c_model =
    wells_glm_model(switched, hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic, educ4))
# wells_data-wells_dae_inter_model
const dae_inter_model = wells_glm_model(
    switched,
    hcat(
        c_dist100,
        c_arsenic,
        c_educ4,
        c_dist100 .* c_arsenic,
        c_dist100 .* c_educ4,
        c_arsenic .* c_educ4,
    ),
)
# wells_data-wells_daae_c_model
const daae_c_model = wells_glm_model(
    switched,
    hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic, assoc, educ4),
)
# wells_data-wells_interaction_model
const interaction_model =
    wells_glm_model(switched, hcat(dist100, arsenic, dist100 .* arsenic))
# wells_data-wells_interaction_c_model
const interaction_c_model =
    wells_glm_model(switched, hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic))

end
