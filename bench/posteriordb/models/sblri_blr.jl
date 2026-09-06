# PosteriorDB: sblri-blr
module SblriBLR

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using LinearAlgebra: I
using Wren: Wren, @wren, @observe
using ProbabilityMeasures: ProbabilityMeasures as PM

using ..PMProducts: ProductNormal

struct _FixedData{T}
    value::T
end

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "sblri-blr")))
end

# ── DynamicPPL ──

@model function dppl_sblri_blr(X, y, D)
    X = X.value
    y = y.value
    beta ~ product_distribution(Fill(Normal(0, 10), D))
    sigma ~ truncated(Normal(0, 10); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(y))),
        y .- X * beta,
    )
end

const turing_model = dppl_sblri_blr(_FixedData(data["X"]), _FixedData(data["y"]), data["D"])

# ── Wren ──

@wren function sblri_blr(y, X, prior_mean)
    beta ~ PM.MvNormal(prior_mean, 10.0 * I)
    sigma ~ PM.Exponential(10.0) # substitutes truncated(Normal(0, 10); lower=0)
    @observe y ~ ProductNormal(X * beta, sigma)
    return nothing
end

const wren_model = let
    X = data["X"]
    sblri_blr(data["y"], X, zeros(data["D"]))
end

end
