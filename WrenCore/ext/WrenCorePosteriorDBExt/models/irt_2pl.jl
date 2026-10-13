# PosteriorDB: irt_2pl-irt_2pl
module IRT2PL

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "irt_2pl-irt_2pl")))
end

function irt_2pl_logp(y, theta, a, b)
    total = zero(theta[1] + a[1] + b[1])
    for j in axes(y, 2), i in axes(y, 1)
        total += logdensityof(WD.BernoulliLogit(a[i] * (theta[j] - b[i])), y[i, j])
    end
    return total
end

@wren function irt_2pl(y, I, J)
    sigma_theta ~ WD.HalfCauchy(0.0, 2.0)
    @iid theta ~ WD.Normal(0.0, sigma_theta) size=J
    sigma_a ~ WD.HalfCauchy(0.0, 2.0)
    @iid a ~ WD.LogNormal(0.0, sigma_a) size=I
    mu_b ~ WD.Normal(0.0, 5.0)
    sigma_b ~ WD.HalfCauchy(0.0, 2.0)
    @iid b ~ WD.Normal(mu_b, sigma_b) size=I
    @addloglikelihood irt_2pl_logp(y, theta, a, b)
    return nothing
end

const wren_model = irt_2pl(data["y"], data["I"], data["J"])

end
