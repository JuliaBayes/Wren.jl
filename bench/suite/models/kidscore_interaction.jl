# PosteriorDB: kidiq-kidscore_interaction
# Deviations: beta prior is Flat() in PDB, replaced with MvNormal(0, 100I) as WrenDists
# doesn't have improper priors.
module KidscoreInteraction

using PosteriorDB
using Wren: Wren, @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kidiq-kidscore_interaction")),
    )
end

@wren function kidscore_interaction(kid_score, X)
    @iid beta ~ WD.Normal(0.0, 10.0) size=size(X, 2)
    sigma ~ WD.HalfCauchy(0.0, 2.5)
    mu = X * beta
    @batch for i in eachindex(kid_score)
        @observe kid_score[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const wren_model = let
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    X = hcat(ones(length(mom_hs)), mom_hs, mom_iq, mom_hs .* mom_iq)
    kidscore_interaction(data["kid_score"], X)
end

end
