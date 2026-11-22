# PosteriorDB: eight_schools-eight_schools_centered and
# eight_schools-eight_schools_noncentered. Both use the same dataset.
module EightSchools

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "eight_schools-eight_schools_centered"),
        ),
    )
end

@wren function eight_schools_centered(y, sigma, J)
    mu ~ WD.Normal(0.0, 5.0)
    tau ~ WD.HalfCauchy(0.0, 5.0)
    @iid theta ~ WD.Normal(mu, tau) size=J
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(theta[i], sigma[i])
    end
    return nothing
end

@wren function eight_schools_noncentered(y, sigma, J)
    @iid theta_trans ~ WD.Normal(0.0, 1.0) size=J
    mu ~ WD.Normal(0.0, 5.0)
    tau ~ WD.HalfCauchy(0.0, 5.0)
    theta = theta_trans .* tau .+ mu
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(theta[i], sigma[i])
    end
    return nothing
end

const centered = eight_schools_centered(data["y"], data["sigma"], data["J"])
const noncentered = eight_schools_noncentered(data["y"], data["sigma"], data["J"])

end
