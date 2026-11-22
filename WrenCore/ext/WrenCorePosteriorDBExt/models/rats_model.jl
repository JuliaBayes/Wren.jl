# PosteriorDB: rats_data-rats_model
module RatsModel

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "rats_data-rats_model")),
    )
end

@wren function rats_model(rat, x, y, xbar, N)
    mu_alpha ~ WD.Normal(0.0, 100.0)
    mu_beta ~ WD.Normal(0.0, 100.0)
    # The three scales have no prior in the Stan source, i.e. they are flat on (0, ∞).
    sigma_y ~ WD.PositiveFlat()
    sigma_alpha ~ WD.PositiveFlat()
    sigma_beta ~ WD.PositiveFlat()
    @iid alpha ~ WD.Normal(mu_alpha, sigma_alpha) size=N
    @iid beta ~ WD.Normal(mu_beta, sigma_beta) size=N
    mu = alpha[rat] .+ beta[rat] .* (x .- xbar)
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const model = rats_model(data["rat"], data["x"], data["y"], data["xbar"], data["N"])

end
