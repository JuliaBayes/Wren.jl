# PosteriorDB: nes1972-nes through nes2000-nes
module NESCommon

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

@wren function nes(partyid7, X)
    @iid beta ~ WD.Flat() size=9
    log_sigma ~ WD.Flat()
    sigma = exp(log_sigma)
    @addloglikelihood log_sigma
    mu = X * beta
    @batch for i in eachindex(partyid7)
        @observe partyid7[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

function fixture(year)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "nes$(year)-nes",
    )))
    age = data["age_discrete"]
    X = hcat(ones(data["N"]), data["real_ideo"], data["race_adj"],
        age .== 2, age .== 3, age .== 4, data["educ1"], data["gender"],
        data["income"])
    return nes(data["partyid7"], X)
end

end

module NES1972
using ..NESCommon: fixture
const wren_model = fixture(1972)
end
module NES1976
using ..NESCommon: fixture
const wren_model = fixture(1976)
end
module NES1980
using ..NESCommon: fixture
const wren_model = fixture(1980)
end
module NES1984
using ..NESCommon: fixture
const wren_model = fixture(1984)
end
module NES1988
using ..NESCommon: fixture
const wren_model = fixture(1988)
end
module NES1992
using ..NESCommon: fixture
const wren_model = fixture(1992)
end
module NES1996
using ..NESCommon: fixture
const wren_model = fixture(1996)
end
module NES2000
using ..NESCommon: fixture
const wren_model = fixture(2000)
end
