# PosteriorDB: radon_all-radon_county_intercept and
# radon_mn-radon_county_intercept. The same model on two datasets.
module RadonCountyIntercept

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

@wren function radon_county_intercept(J, county_idx, floor_measure, log_radon)
    @iid alpha ~ WD.Normal(0.0, 10.0) size=J
    beta ~ WD.Normal(0.0, 10.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    mu = alpha[county_idx] .+ beta .* floor_measure
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

function load_model(name)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    return radon_county_intercept(
        data["J"],
        data["county_idx"],
        data["floor_measure"],
        data["log_radon"],
    )
end

const radon_all = load_model("radon_all-radon_county_intercept")
const radon_mn = load_model("radon_mn-radon_county_intercept")

end
