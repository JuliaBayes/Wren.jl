# PosteriorDB: kilpisjarvi_mod-kilpisjarvi
module Kilpisjarvi

using PosteriorDB
using WrenCore: @wren, @observe, @batch
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kilpisjarvi_mod-kilpisjarvi")),
    )
end

@wren function kilpisjarvi_model(x, y, pmualpha, psalpha, pmubeta, psbeta)
    alpha ~ WD.Normal(pmualpha, psalpha)
    beta ~ WD.Normal(pmubeta, psbeta)
    sigma ~ WD.PositiveFlat()
    mu = alpha .+ beta .* x
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const model = kilpisjarvi_model(
    data["x"],
    data["y"],
    data["pmualpha"],
    data["psalpha"],
    data["pmubeta"],
    data["psbeta"],
)

end
