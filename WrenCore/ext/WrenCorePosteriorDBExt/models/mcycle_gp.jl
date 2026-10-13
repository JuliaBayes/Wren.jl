# PosteriorDB: mcycle_gp-accel_gp
module McycleGP

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "mcycle_gp-accel_gp",
    )))
end

function gp_basis_at(X, frequencies, z, sdgp, lscale, n)
    total = zero(sdgp + z[1])
    for m in eachindex(z)
        total += X[n, m] * z[m] * exp(-0.25 * lscale^2 * frequencies[m]^2)
    end
    # Square root of the one-dimensional exponentiated quadratic spectral density.
    return sdgp * sqrt(sqrt(2 * pi) * lscale) * total
end

function accel_loglikelihood(Y, X, frequencies, Xsigma, frequencies_sigma,
        intercept, sdgp, lscale, zgp, intercept_sigma, sdgp_sigma,
        lscale_sigma, zgp_sigma)
    total = zero(intercept + intercept_sigma)
    for n in eachindex(Y)
        mu = intercept + gp_basis_at(X, frequencies, zgp, sdgp, lscale, n)
        log_sigma = intercept_sigma +
            gp_basis_at(Xsigma, frequencies_sigma, zgp_sigma,
                sdgp_sigma, lscale_sigma, n)
        total += logdensityof(WD.Normal(mu, exp(log_sigma)), Y[n])
    end
    return total
end

@wren function accel_gp(Y, X, frequencies, Xsigma, frequencies_sigma)
    intercept ~ WD.TDist(3.0, -13.0, 36.0)
    raw_sdgp ~ WD.Flat()
    raw_lscale ~ WD.Flat()
    @iid zgp ~ WD.Normal(0.0, 1.0) size=40
    intercept_sigma ~ WD.TDist(3.0, 0.0, 10.0)
    raw_sdgp_sigma ~ WD.Flat()
    raw_lscale_sigma ~ WD.Flat()
    @iid zgp_sigma ~ WD.Normal(0.0, 1.0) size=20

    sdgp = exp(raw_sdgp)
    lscale = exp(raw_lscale)
    sdgp_sigma = exp(raw_sdgp_sigma)
    lscale_sigma = exp(raw_lscale_sigma)
    # Stan's truncated Student-t prior contributes log(2) on each positive
    # marginal scale. The Gamma precision terms are exactly the Stan inverse-
    # gamma length-scale priors after its log support transform.
    inverse_gamma_shape = 1.124909
    inverse_gamma_rate = 0.0177
    @addloglikelihood logdensityof(WD.TDist(3.0, 0.0, 36.0), sdgp) +
        log(2.0) + raw_sdgp +
        logdensityof(WD.Gamma(inverse_gamma_shape, 1 / inverse_gamma_rate),
            exp(-raw_lscale)) - raw_lscale +
        logdensityof(WD.TDist(3.0, 0.0, 36.0), sdgp_sigma) +
        log(2.0) + raw_sdgp_sigma +
        logdensityof(WD.Gamma(inverse_gamma_shape, 1 / inverse_gamma_rate),
            exp(-raw_lscale_sigma)) - raw_lscale_sigma
    @addloglikelihood accel_loglikelihood(Y, X, frequencies, Xsigma,
        frequencies_sigma, intercept, sdgp, lscale, zgp, intercept_sigma,
        sdgp_sigma, lscale_sigma, zgp_sigma)
    return nothing
end

const wren_model = let
    data["N"] == 133 && data["NBgp_1"] == 40 && data["NBgp_sigma_1"] == 20 &&
        data["Dgp_1"] == 1 && data["Dgp_sigma_1"] == 1 &&
        data["Kgp_1"] == 1 && data["Kgp_sigma_1"] == 1 &&
        data["prior_only"] == 0 || error("unexpected public motorcycle GP fixture")
    accel_gp(data["Y"], data["Xgp_1"], vec(data["slambda_1"]),
        data["Xgp_sigma_1"], vec(data["slambda_sigma_1"]))
end

end
