# PosteriorDB: nes1972-nes through nes2000-nes. All eight use the same Stan model on a
# different year's survey data.
module NES

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

@wren function nes_base(partyid7, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    sigma ~ WD.PositiveFlat()
    mu = X * beta
    @batch for i in eachindex(partyid7)
        @observe partyid7[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

function nes_model(year)
    pdb = PosteriorDB.database()
    data =
        PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "nes$(year)-nes")))
    age = data["age_discrete"]
    X = hcat(
        ones(data["N"]),
        data["real_ideo"],
        data["race_adj"],
        age .== 2,
        age .== 3,
        age .== 4,
        data["educ1"],
        data["gender"],
        data["income"],
    )
    return nes_base(data["partyid7"], X)
end

const nes1972 = nes_model(1972)
const nes1976 = nes_model(1976)
const nes1980 = nes_model(1980)
const nes1984 = nes_model(1984)
const nes1988 = nes_model(1988)
const nes1992 = nes_model(1992)
const nes1996 = nes_model(1996)
const nes2000 = nes_model(2000)

end
