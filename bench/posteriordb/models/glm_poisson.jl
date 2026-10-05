# PosteriorDB: GLM_Poisson_Data-GLM_Poisson_model
module GLMPoisson

using PosteriorDB
using DynamicPPL, Distributions
using Wren: Wren, @wren, @observe, @batch
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLM_Poisson_Data-GLM_Poisson_model"),
        ),
    )
end

# ── DynamicPPL ──

@model function pdb_GLM_Poisson_model(n, C, year, year_squared, year_cubed)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 10)
    beta3 ~ Uniform(-10, 10)
    log_lambda = alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(x -> Poisson(exp(x)), log_lambda)),
        C,
    )
end

const turing_model = let
    year = data["year"]
    pdb_GLM_Poisson_model(data["n"], data["C"], year, year .^ 2, year .^ 3)
end

# ── Wren ──

@wren function glm_poisson_model(counts, year, year_squared, year_cubed)
    alpha ~ WD.Uniform(-20.0, 20.0)
    beta1 ~ WD.Uniform(-10.0, 10.0)
    beta2 ~ WD.Uniform(-10.0, 10.0)
    beta3 ~ WD.Uniform(-10.0, 10.0)
    log_lambda = alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed
    @batch for i in eachindex(counts)
        @observe counts[i] ~ WD.PoissonLog(log_lambda[i])
    end
    return nothing
end

const wren_model = let
    counts = data["C"]
    year = data["year"]
    glm_poisson_model(counts, year, year .^ 2, year .^ 3)
end

end
