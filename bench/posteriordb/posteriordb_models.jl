# Wren translations of ten PosteriorDB posteriors.
#
# ProbabilityMeasures does not cover every prior family the PosteriorDB catalog
# uses, so several priors here are substituted. A substituted model has a
# different posterior from the Stan program it is named after: the timings are
# comparable, the log-density values are not. Every substitution is marked with a
# `# substitutes ...` comment at the tilde statement, and the substitution rules
# are:
#
#   truncated(Cauchy(0, s); lower=0)  ->  Exponential(s)
#   truncated(Normal(0, s); lower=0)  ->  Exponential(s)
#   Flat()                            ->  Normal(0, 100)
#
# Two rewrites are exact rather than substitutions, and are marked as such:
# `Beta(1, 1)` is `Uniform(0, 1)`, and a binomial likelihood on the logit scale
# is `ProductBinomialLogit`.
#
# Structural notes:
#   - Observation blocks use the measures in `pmproducts.jl`. They evaluate as
#     array arithmetic and a reduction, which is what Reactant can trace;
#     `@plate` lowers to a Wren measure that reads its arguments one element at a
#     time, and Reactant rejects it.
#   - Prior means, design matrices, lag matrices and the data-only terms of the
#     count likelihoods are built in `make_model`, so an evaluation allocates
#     nothing that does not depend on the parameters.

using LogExpFunctions: log1pexp, logaddexp, logit
using ProbabilityMeasures: ProbabilityMeasures as PM, logdensityof
using Wren: Wren, @wren, @observe, @addloglikelihood, I

include(joinpath(@__DIR__, "pmproducts.jl"))
using .PMProducts:
    ProductNormal,
    ProductPoisson,
    ProductBinomialLogit,
    DenseMvNormal,
    poisson_logconst,
    binomial_logconst

const POSTERIOR_NAMES = [
    "Rate_1_data-Rate_1_model",
    "GLM_Binomial_data-GLM_Binomial_model",
    "gp_pois_regr-gp_regr",
    "GLM_Poisson_Data-GLM_Poisson_model",
    "kidiq-kidscore_interaction",
    "sblri-blr",
    "arK-arK",
    "eight_schools-eight_schools_centered",
    "GLMM_Poisson_data-GLMM_Poisson_model",
    "Mh_data-Mh_model",
]

# ── Rate_1_model: one binomial proportion ──

@wren function rate_1_model(k, n)
    theta ~ PM.Uniform(0.0, 1.0) # Beta(1, 1)
    @observe k ~ PM.Binomial(n, theta)
    return nothing
end

function make_model(::Val{Symbol("Rate_1_data-Rate_1_model")}, data)
    return rate_1_model(Int(data["k"]), Int(data["n"]))
end

# ── GLM_Binomial_model: binomial regression on a cubic year trend ──

@wren function glm_binomial_model(counts, trials, year, year_squared, logconst)
    alpha ~ PM.Normal(0.0, 100.0)
    beta1 ~ PM.Normal(0.0, 100.0)
    beta2 ~ PM.Normal(0.0, 100.0)
    logit_p = alpha .+ beta1 .* year .+ beta2 .* year_squared
    # BinomialLogit
    @observe counts ~ ProductBinomialLogit(trials, logit_p, logconst)
    return nothing
end

function make_model(::Val{Symbol("GLM_Binomial_data-GLM_Binomial_model")}, data)
    counts = Float64.(data["C"])
    trials = Float64.(data["N"])
    year = Float64.(data["year"])
    return glm_binomial_model(
        counts, trials, year, year .^ 2, binomial_logconst(trials, counts)
    )
end

# ── gp_regr: Gaussian process regression with a squared-exponential kernel ──

@wren function gp_regr(y, dist_sq, prior_mean)
    rho ~ PM.Gamma(25.0, 1 / 4)
    alpha ~ PM.Exponential(2.0) # substitutes truncated(Normal(0, 2); lower=0)
    sigma ~ PM.Exponential(1.0) # substitutes truncated(Normal(0, 1); lower=0)
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + sigma * I
    @observe y ~ DenseMvNormal(prior_mean, covariance)
    return nothing
end

function make_model(::Val{Symbol("gp_pois_regr-gp_regr")}, data)
    x = Float64.(data["x"])
    y = Float64.(data["y"])
    return gp_regr(y, (x .- x') .^ 2, zeros(length(y)))
end

# ── GLM_Poisson_model: Poisson regression on a cubic year trend ──

@wren function glm_poisson_model(counts, year, year_squared, year_cubed, logconst)
    alpha ~ PM.Uniform(-20.0, 20.0)
    beta1 ~ PM.Uniform(-10.0, 10.0)
    beta2 ~ PM.Uniform(-10.0, 10.0)
    beta3 ~ PM.Uniform(-10.0, 10.0)
    log_lambda = alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed
    @observe counts ~ ProductPoisson(exp.(log_lambda), logconst)
    return nothing
end

function make_model(::Val{Symbol("GLM_Poisson_Data-GLM_Poisson_model")}, data)
    counts = Float64.(data["C"])
    year = Float64.(data["year"])
    return glm_poisson_model(counts, year, year .^ 2, year .^ 3, poisson_logconst(counts))
end

# ── kidscore_interaction: linear regression with an interaction term ──

@wren function kidscore_interaction(kid_score, X, prior_mean)
    beta ~ PM.MvNormal(prior_mean, 100.0 * I) # substitutes Flat()
    sigma ~ PM.Exponential(2.5) # substitutes truncated(Cauchy(0, 2.5); lower=0)
    @observe kid_score ~ ProductNormal(X * beta, sigma)
    return nothing
end

function make_model(::Val{Symbol("kidiq-kidscore_interaction")}, data)
    mom_hs = Float64.(data["mom_hs"])
    mom_iq = Float64.(data["mom_iq"])
    X = hcat(ones(length(mom_hs)), mom_hs, mom_iq, mom_hs .* mom_iq)
    return kidscore_interaction(Float64.(data["kid_score"]), X, zeros(size(X, 2)))
end

# ── blr: Bayesian linear regression on a dense design matrix ──

@wren function sblri_blr(y, X, prior_mean)
    beta ~ PM.MvNormal(prior_mean, 10.0 * I)
    sigma ~ PM.Exponential(10.0) # substitutes truncated(Normal(0, 10); lower=0)
    @observe y ~ ProductNormal(X * beta, sigma)
    return nothing
end

function make_model(::Val{Symbol("sblri-blr")}, data)
    X = Float64.(data["X"])
    return sblri_blr(Float64.(data["y"]), X, zeros(Int(data["D"])))
end

# ── arK: autoregressive model of order K ──

@wren function ark_model(y, lags, prior_mean)
    alpha ~ PM.Normal(0.0, 10.0)
    beta ~ PM.MvNormal(prior_mean, 10.0 * I)
    sigma ~ PM.Exponential(2.5) # substitutes truncated(Cauchy(0, 2.5); lower=0)
    @observe y ~ ProductNormal(alpha .+ lags * beta, sigma)
    return nothing
end

function make_model(::Val{Symbol("arK-arK")}, data)
    K = Int(data["K"])
    T = Int(data["T"])
    series = Float64.(data["y"])
    # Row t of `lags` holds the K values preceding observation t.
    lags = [series[t - k] for t in (K + 1):T, k in 1:K]
    return ark_model(series[(K + 1):T], lags, zeros(K))
end

# ── eight_schools_centered: the centred hierarchical normal model ──

@wren function eight_schools_centered(y, sigma, J)
    mu ~ PM.Normal(0.0, 5.0)
    tau ~ PM.Exponential(5.0) # substitutes truncated(Cauchy(0, 5); lower=0)
    theta ~ PM.MvNormal(fill(mu, J), tau * I)
    @observe y ~ ProductNormal(theta, sigma)
    return nothing
end

function make_model(::Val{Symbol("eight_schools-eight_schools_centered")}, data)
    return eight_schools_centered(
        Float64.(data["y"]), Float64.(data["sigma"]), Int(data["J"])
    )
end

# ── GLMM_Poisson_model: Poisson regression with one random effect per year ──

@wren function glmm_poisson_model(
    counts, year, year_squared, year_cubed, prior_mean, logconst
)
    alpha ~ PM.Uniform(-20.0, 20.0)
    beta1 ~ PM.Uniform(-10.0, 10.0)
    beta2 ~ PM.Uniform(-10.0, 20.0)
    beta3 ~ PM.Uniform(-10.0, 10.0)
    sigma ~ PM.Uniform(0.0, 5.0)
    eps ~ PM.MvNormal(prior_mean, 100.0 * I) # substitutes Flat()
    log_lambda =
        alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed .+ eps
    @observe counts ~ ProductPoisson(exp.(log_lambda), logconst)
    @addloglikelihood logdensityof(PM.MvNormal(prior_mean, sigma * I), eps)
    return nothing
end

function make_model(::Val{Symbol("GLMM_Poisson_data-GLMM_Poisson_model")}, data)
    counts = Float64.(data["C"])
    year = Float64.(data["year"])
    return glmm_poisson_model(
        counts, year, year .^ 2, year .^ 3, zeros(Int(data["n"])), poisson_logconst(counts)
    )
end

# ── Mh_model: capture-recapture with individual heterogeneity ──
#
# The population is augmented to `M` individuals. Individuals seen at least once
# contribute a binomial capture record; the rest contribute a mixture over being
# absent from the population and being present but never captured.

@wren function mh_model(captures, detected, undetected, trials, prior_mean, logconst)
    omega ~ PM.Uniform(0.0, 1.0)
    mean_p ~ PM.Uniform(0.0, 1.0)
    sigma ~ PM.Uniform(0.0, 5.0)
    eps_raw ~ PM.MvNormal(prior_mean, 1.0 * I)
    logit_p = logit(mean_p) .+ sigma .* eps_raw
    # BinomialLogit
    @observe captures ~ ProductBinomialLogit(trials, logit_p[detected], logconst)
    @addloglikelihood length(detected) * log(omega) + sum(
        logaddexp.(log(omega) .- trials .* log1pexp.(logit_p[undetected]), log1p(-omega))
    )
    return nothing
end

function make_model(::Val{Symbol("Mh_data-Mh_model")}, data)
    M = Int(data["M"])
    trials = Float64(data["T"])
    y = Float64.(data["y"])
    detected = findall(>(0), y)
    undetected = findall(==(0), y)
    captures = y[detected]
    return mh_model(
        captures,
        detected,
        undetected,
        trials,
        zeros(M),
        binomial_logconst(trials, captures),
    )
end
