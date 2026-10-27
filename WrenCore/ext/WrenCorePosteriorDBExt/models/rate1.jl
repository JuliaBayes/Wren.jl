# PosteriorDB: Rate_1_data-Rate_1_model
module Rate1

using PosteriorDB
using WrenCore: @wren, @observe
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Rate_1_data-Rate_1_model")),
    )
end

@wren function rate_1_model(k, n)
    theta ~ WD.Uniform(0.0, 1.0)
    @observe k ~ WD.Binomial(n, theta)
    return nothing
end

const wren_model = rate_1_model(data["k"], data["n"])

end
