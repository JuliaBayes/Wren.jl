# PosteriorDB: radon_all-radon_variable_slope_centered and
# radon_mn-radon_variable_slope_centered. The same model on two datasets.
module RadonVariableSlopeCentered

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

@wren function radon_variable_slope_centered(J, county_idx, floor_measure, log_radon)
    alpha ~ WD.Normal(0.0, 10.0)
    @iid beta ~ WD.Flat() size=J
    mu_beta ~ WD.Normal(0.0, 10.0)
    sigma_beta ~ WD.HalfNormal(0.0, 1.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    @observe @iid beta ~ WD.Normal(mu_beta, sigma_beta) size=J
    mu = alpha .+ beta[county_idx] .* floor_measure
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

function load_model(name)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    return radon_variable_slope_centered(
        data["J"],
        data["county_idx"],
        data["floor_measure"],
        data["log_radon"],
    )
end

const radon_all = load_model("radon_all-radon_variable_slope_centered")
const radon_mn = load_model("radon_mn-radon_variable_slope_centered")

end
