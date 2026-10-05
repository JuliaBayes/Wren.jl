# https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/arK.stan

module ArK

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill
using Wren: Wren, @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "arK-arK")))
end

# ── DynamicPPL ──

@model function arK_arK(y, lags, K)
    alpha ~ Normal(0, 10)
    beta ~ product_distribution(Fill(Normal(0, 10), K))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    mu = alpha .+ lags * beta
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), length(y))),
        y .- mu,
    )
end

const turing_model = let
    K = data["K"]
    T = data["T"]
    series = data["y"]
    lags = [series[t-k] for t in (K+1):T, k in 1:K]
    arK_arK(series[(K+1):T], lags, K)
end

# ── Wren ──

@wren function ark_model(y, lags, K)
    alpha ~ WD.Normal(0.0, 10.0)
    @iid beta ~ WD.Normal(0.0, 10.0) size=K
    sigma ~ WD.HalfCauchy(0.0, 2.5)
    mu = alpha .+ lags * beta
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
end

const wren_model = let
    K = data["K"]
    T = data["T"]
    series = data["y"]
    # Row t of `lags` holds the K values preceding observation t.
    lags = [series[t-k] for t in (K+1):T, k in 1:K]
    ark_model(series[(K+1):T], lags, K)
end

end
