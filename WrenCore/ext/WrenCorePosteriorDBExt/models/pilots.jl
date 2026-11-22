# PosteriorDB: pilots-pilots
module Pilots

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "pilots-pilots")))
end

@wren function pilots_model(group_id, scenario_id, y, n_groups, n_scenarios)
    mu_a ~ WD.Normal(0.0, 1.0)
    mu_b ~ WD.Normal(0.0, 1.0)
    # In the Stan source, the three `sigma`s are declared with bounds (0, 100) but have no
    # prior, so they are flat on (0, 100). WrenDists technically has TruncatedFlat, but we
    # use `Uniform(0, 100)` since it's more readable. The log-density is the same up to a
    # constant.
    sigma_a ~ WD.Uniform(0.0, 100.0)
    sigma_b ~ WD.Uniform(0.0, 100.0)
    sigma_y ~ WD.Uniform(0.0, 100.0)
    @iid a ~ WD.Normal(10 * mu_a, sigma_a) size=n_groups
    @iid b ~ WD.Normal(10 * mu_b, sigma_b) size=n_scenarios
    @track y_hat = a[group_id] .+ b[scenario_id]
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(y_hat[i], sigma_y)
    end
    return nothing
end

const model = pilots_model(
    data["group_id"],
    data["scenario_id"],
    data["y"],
    data["n_groups"],
    data["n_scenarios"],
)

end
