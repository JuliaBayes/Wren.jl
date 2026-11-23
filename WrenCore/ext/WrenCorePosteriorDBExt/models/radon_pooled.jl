# PosteriorDB: radon_all-radon_pooled and radon_mn-radon_pooled. The same model on two
# datasets.
module RadonPooled

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

@wren function radon_pooled_model(log_radon, floor_measure)
    alpha ~ WD.Normal(0.0, 10.0)
    beta ~ WD.Normal(0.0, 10.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    mu = alpha .+ beta .* floor_measure
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

function load_model(name)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    return radon_pooled_model(data["log_radon"], data["floor_measure"])
end

const radon_all = load_model("radon_all-radon_pooled")
const radon_mn = load_model("radon_mn-radon_pooled")

end
