# PosteriorDB: wells_data-wells_dae_inter_model
module WellsDaeInter

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using Statistics: mean

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "wells_data-wells_dae_inter_model")),
    )
end

@model function pdb_wells_dae_inter_model(switched, X)
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), size(X, 2)))
    logit_p = alpha .+ X * beta
    for i in eachindex(switched)
        switched[i] ~ BernoulliLogit(logit_p[i])
    end
end

const turing_model = let
    c_dist100 = (data["dist"] .- mean(data["dist"])) ./ 100.0
    c_arsenic = data["arsenic"] .- mean(data["arsenic"])
    c_educ4 = (data["educ"] .- mean(data["educ"])) ./ 4.0
    X = hcat(
        c_dist100,
        c_arsenic,
        c_educ4,
        c_dist100 .* c_arsenic, # da_inter
        c_dist100 .* c_educ4,   # de_inter
        c_arsenic .* c_educ4,   # ae_inter
    )
    pdb_wells_dae_inter_model(data["switched"], X)
end

end
