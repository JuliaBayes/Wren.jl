# PosteriorDB: sat-hier_2pl
module SAT2PL

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "sat-hier_2pl",
    )))
end

function sat_logdensity(y, ii, jj, theta, xi1, xi2, mu1, mu2,
        tau1, tau2, rho)
    total = zero(mu1 + mu2 + tau1 + tau2 + rho)
    residual_scale = sqrt(1 - rho^2)
    log_normalizer = log(2 * pi) + log(tau1) + log(tau2) +
        log(residual_scale)
    for i in eachindex(xi1)
        z1 = (xi1[i] - mu1) / tau1
        z2 = ((xi2[i] - mu2) / tau2 - rho * z1) / residual_scale
        total += -log_normalizer - 0.5 * (z1^2 + z2^2)
    end
    for n in eachindex(y)
        eta = exp(xi1[ii[n]]) * (theta[jj[n]] - xi2[ii[n]])
        total += logdensityof(WD.BernoulliLogit(eta), y[n])
    end
    return total
end

@wren function hier_2pl(y, ii, jj, I, J)
    @iid theta ~ WD.Normal(0.0, 1.0) size=J
    @iid xi1 ~ WD.Flat() size=I
    @iid xi2 ~ WD.Flat() size=I
    mu1 ~ WD.Normal(0.0, 1.0)
    mu2 ~ WD.Normal(0.0, 5.0)
    raw_tau1 ~ WD.Flat()
    raw_tau2 ~ WD.Flat()
    raw_corr ~ WD.Flat()

    tau1 = exp(raw_tau1)
    tau2 = exp(raw_tau2)
    rho = tanh(raw_corr)
    # K=2 LKJ-Cholesky(4) is the transformed Beta(4,4) correlation prior.
    # Exponential(10) uses the scale corresponding to Stan's rate 0.1.
    @addloglikelihood logdensityof(WD.Exponential(10.0), tau1) + raw_tau1 +
        logdensityof(WD.Exponential(10.0), tau2) + raw_tau2 +
        logdensityof(WD.Beta(4.0, 4.0), (rho + 1) / 2) - log(2.0) +
        log1p(-rho^2)
    @addloglikelihood sat_logdensity(y, ii, jj, theta, xi1, xi2,
        mu1, mu2, tau1, tau2, rho)
    return nothing
end

const wren_model = let
    data["I"] == 32 && data["J"] == 600 && data["N"] == 19200 ||
        error("unexpected public SAT fixture")
    hier_2pl(data["y"], data["ii"], data["jj"], data["I"], data["J"])
end

end
