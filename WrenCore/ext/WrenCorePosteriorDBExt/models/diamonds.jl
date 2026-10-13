# PosteriorDB: diamonds-diamonds
module Diamonds

using PosteriorDB
using Statistics: mean
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "diamonds-diamonds")))
end

function diamonds_logp(y, Xc, b, intercept, sigma)
    total = zero(b[1] + intercept + sigma)
    for i in eachindex(y)
        prediction = intercept
        for j in eachindex(b)
            prediction += Xc[i, j] * b[j]
        end
        total += logdensityof(WD.Normal(prediction, sigma), y[i])
    end
    return total
end

@wren function diamonds(y, Xc, prior_only)
    @iid b ~ WD.Normal(0.0, 1.0) size=size(Xc, 2)
    Intercept ~ WD.TDist(3.0, 8.0, 10.0)
    sigma ~ WD.HalfTDist(3.0, 0.0, 10.0)
    if prior_only == 0
        @addloglikelihood diamonds_logp(y, Xc, b, Intercept, sigma)
    end
    return nothing
end

const wren_model = let
    X = data["X"]
    Xc = X[:, 2:end] .- mean(X[:, 2:end]; dims=1)
    diamonds(data["Y"], Xc, data["prior_only"])
end

end
