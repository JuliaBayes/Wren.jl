# PosteriorDB: eight_schools-eight_schools_centered
module EightSchoolsCentered

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using Wren: Wren, @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "eight_schools-eight_schools_centered"),
        ),
    )
end

# ── DynamicPPL ──

@model function dppl_eight_schools_centered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ product_distribution(Fill(Normal(mu, tau), J))
    for i in 1:J
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

const turing_model = dppl_eight_schools_centered(data["J"], data["y"], data["sigma"])

# ── Wren ──

@wren function eight_schools_centered(y, sigma, J)
    mu ~ WD.Normal(0.0, 5.0)
    tau ~ WD.HalfCauchy(0.0, 5.0)
    @iid theta ~ WD.Normal(mu, tau) size=J
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(theta[i], sigma[i])
    end
    return nothing
end

const wren_model = eight_schools_centered(data["y"], data["sigma"], data["J"])

end
