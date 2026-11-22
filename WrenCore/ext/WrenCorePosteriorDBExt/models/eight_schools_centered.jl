# PosteriorDB: eight_schools-eight_schools_centered
module EightSchoolsCentered

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
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

const model = eight_schools_centered(data["y"], data["sigma"], data["J"])

end
