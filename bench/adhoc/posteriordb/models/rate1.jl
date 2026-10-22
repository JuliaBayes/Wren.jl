# PosteriorDB: Rate_1_data-Rate_1_model
module Rate1

using PosteriorDB
using DynamicPPL, Distributions
using Wren: Wren, @wren, @observe
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Rate_1_data-Rate_1_model")),
    )
end

# ── DynamicPPL ──

@model function pdb_Rate_1_model(n, k)
    theta ~ Beta(1, 1)
    return k ~ Binomial(n, theta)
end

const turing_model = pdb_Rate_1_model(data["n"], data["k"])

# ── Wren ──

@wren function rate_1_model(k, n)
    theta ~ WD.Uniform(0.0, 1.0)
    @observe k ~ WD.Binomial(n, theta)
    return nothing
end

const wren_model = rate_1_model(data["k"], data["n"])

end
