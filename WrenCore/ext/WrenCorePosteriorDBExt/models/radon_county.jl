# PosteriorDB: radon_mod-radon_county
module RadonCounty

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "radon_mod-radon_county",
    )))
end

@wren function radon_county(county, y, J)
    mu_a ~ WD.Normal(0.0, 1.0)
    raw_sigma_a ~ WD.Flat()
    raw_sigma_y ~ WD.Flat()
    ratio_a = inv(1 + exp(-raw_sigma_a))
    ratio_y = inv(1 + exp(-raw_sigma_y))
    sigma_a = 100 * ratio_a
    sigma_y = 100 * ratio_y
    # Both Stan scales are constrained to (0, 100) without a prior factor.
    @addloglikelihood 2 * log(100.0) + log(ratio_a) + log1p(-ratio_a) +
        log(ratio_y) + log1p(-ratio_y)
    @iid a ~ WD.Normal(mu_a, sigma_a) size=J
    y_hat = a[county]
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(y_hat[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_county(data["county"], data["y"], data["J"])

end
