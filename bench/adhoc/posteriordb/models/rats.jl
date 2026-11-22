# PosteriorDB: rats_data-rats_model
module Rats

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "rats_data-rats_model")),
    )
end

@model function pdb_rats_model(rat, x, y, xbar, N)
    mu_alpha ~ Normal(0, 100)
    mu_beta ~ Normal(0, 100)
    sigma_y ~ FlatPos(0)
    sigma_alpha ~ FlatPos(0)
    sigma_beta ~ FlatPos(0)
    alpha ~ product_distribution(Fill(Normal(mu_alpha, sigma_alpha), N))
    beta ~ product_distribution(Fill(Normal(mu_beta, sigma_beta), N))
    mu = alpha[rat] .+ beta[rat] .* (x .- xbar)
    for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma_y)
    end
end

const turing_model =
    pdb_rats_model(data["rat"], data["x"], data["y"], data["xbar"], data["N"])

end
