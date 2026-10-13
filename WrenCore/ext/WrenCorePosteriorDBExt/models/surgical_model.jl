# PosteriorDB: surgical_data-surgical_model
module SurgicalModel

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "surgical_data-surgical_model",
    )))
end

@wren function surgical_model(r, n)
    mu ~ WD.Normal(0.0, 1000.0)
    # Stan's sigmasq ~ inv_gamma(0.001, 0.001) is exactly equivalent to
    # precision = 1/sigmasq ~ Gamma(shape=0.001, scale=1000).
    precision ~ WD.Gamma(0.001, 1000.0)
    sigma = inv(sqrt(precision))
    @iid b ~ WD.Normal(mu, sigma) size=length(r)
    @batch for i in eachindex(r)
        @observe r[i] ~ WD.BinomialLogit(n[i], b[i])
    end
    return nothing
end

const wren_model = surgical_model(data["r"], data["n"])

end
