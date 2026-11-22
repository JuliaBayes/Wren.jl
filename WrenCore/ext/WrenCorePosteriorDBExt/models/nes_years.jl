# PosteriorDB: nes1972-nes through nes2000-nes. All eight use the same Stan model on a
# different year's survey data.
module NESBase

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
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

end

module NES1972
using ..NESBase: nes_model
const wren_model = nes_model(1972)
end
module NES1976
using ..NESBase: nes_model
const wren_model = nes_model(1976)
end
module NES1980
using ..NESBase: nes_model
const wren_model = nes_model(1980)
end
module NES1984
using ..NESBase: nes_model
const wren_model = nes_model(1984)
end
module NES1988
using ..NESBase: nes_model
const wren_model = nes_model(1988)
end
module NES1992
using ..NESBase: nes_model
const wren_model = nes_model(1992)
end
module NES1996
using ..NESBase: nes_model
const wren_model = nes_model(1996)
end
module NES2000
using ..NESBase: nes_model
const wren_model = nes_model(2000)
end
