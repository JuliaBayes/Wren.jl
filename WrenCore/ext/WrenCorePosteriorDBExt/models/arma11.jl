# PosteriorDB: arma-arma11
module ARMA11

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "arma-arma11")))
end

function arma11_logp(y, mu, phi, theta, sigma)
    prediction = mu + phi * mu
    previous_error = y[1] - prediction
    total = logdensityof(WD.Normal(0.0, sigma), previous_error)
    for t in 2:length(y)
        prediction = mu + phi * y[t - 1] + theta * previous_error
        previous_error = y[t] - prediction
        total += logdensityof(WD.Normal(0.0, sigma), previous_error)
    end
    return total
end

@wren function arma11(y)
    mu ~ WD.Normal(0.0, 10.0)
    phi ~ WD.Normal(0.0, 2.0)
    theta ~ WD.Normal(0.0, 2.0)
    # Stan's positive Cauchy factor differs by the constant log(2).
    sigma ~ WD.HalfCauchy(0.0, 2.5)
    @addloglikelihood arma11_logp(y, mu, phi, theta, sigma)
    return nothing
end

const wren_model = arma11(data["y"])

end
