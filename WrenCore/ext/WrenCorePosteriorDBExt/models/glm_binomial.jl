# PosteriorDB: GLM_Binomial_data-GLM_Binomial_model
module GLMBinomial

using LogExpFunctions: logistic
using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLM_Binomial_data-GLM_Binomial_model"),
        ),
    )
end

@wren function glm_binomial_model(counts, trials, year, year_squared)
    alpha ~ WD.Normal(0.0, 100.0)
    beta1 ~ WD.Normal(0.0, 100.0)
    beta2 ~ WD.Normal(0.0, 100.0)
    @track logit_p = alpha .+ beta1 .* year .+ beta2 .* year_squared
    @batch for i in eachindex(counts)
        @observe counts[i] ~ WD.BinomialLogit(trials[i], logit_p[i])
    end
    @track p = logistic.(logit_p)
    return nothing
end

const model = let
    counts = data["C"]
    trials = data["N"]
    year = data["year"]
    glm_binomial_model(counts, trials, year, year .^ 2)
end

end
