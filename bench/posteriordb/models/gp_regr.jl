# PosteriorDB: gp_pois_regr-gp_regr
module GPRegr

using PosteriorDB
using DynamicPPL, Distributions
using LinearAlgebra: Symmetric, I
using Wren: Wren, @wren, @observe
using ProbabilityMeasures: ProbabilityMeasures as PM

using ..PMProducts: DenseMvNormal

struct _FixedData{T}
    value::T
end

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "gp_pois_regr-gp_regr")),
    )
end

# ── DynamicPPL ──

@model function dppl_gp_regr(dist_sq, y, N)
    dist_sq = dist_sq.value
    y = y.value
    rho ~ Gamma(25, 1 / 4)
    alpha ~ truncated(Normal(0, 2); lower=0)
    sigma ~ truncated(Normal(0, 1); lower=0)
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + sigma * I
    return y ~ MvNormal(zeros(N), Symmetric(covariance))
end

const turing_model = let
    x = data["x"]
    dppl_gp_regr(_FixedData((x .- x') .^ 2), _FixedData(data["y"]), data["N"])
end

# ── Wren ──

@wren function gp_regr(y, dist_sq, prior_mean)
    rho ~ PM.Gamma(25.0, 1 / 4)
    alpha ~ PM.Exponential(2.0) # substitutes truncated(Normal(0, 2); lower=0)
    sigma ~ PM.Exponential(1.0) # substitutes truncated(Normal(0, 1); lower=0)
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + sigma * I
    @observe y ~ DenseMvNormal(prior_mean, covariance)
    return nothing
end

const wren_model = let
    x = data["x"]
    y = data["y"]
    gp_regr(y, (x .- x') .^ 2, zeros(length(y)))
end

end
