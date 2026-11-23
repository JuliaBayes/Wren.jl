# PosteriorDB: sblri-blr
module SblriBLR

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "sblri-blr")))
end

@wren function sblri_blr(y, X, D)
    @iid beta ~ WD.Normal(0.0, 10.0) size=D
    sigma ~ WD.HalfNormal(0.0, 10.0)
    mu = X * beta
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const model = let
    X = data["X"]
    sblri_blr(data["y"], X, data["D"])
end

end
