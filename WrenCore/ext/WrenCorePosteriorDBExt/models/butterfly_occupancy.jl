# PosteriorDB: butterfly-multi_occupancy
module ButterflyOccupancy

using PosteriorDB
using LogExpFunctions: logaddexp, log1pexp
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "butterfly-multi_occupancy",
    )))
end

function occupancy_logdensity(X, K, J, n, S, alpha, beta, omega, rho,
        sigma1, sigma2, uv1, uv2)
    total = zero(alpha + beta + omega + rho + sigma1 + sigma2)
    inv_one_minus_rho_sq = inv(1 - rho^2)
    logdet_term = log(2 * pi) + log(sigma1) + log(sigma2) +
        0.5 * log1p(-rho^2)
    for i in 1:S
        z1 = uv1[i] / sigma1
        z2 = (uv2[i] / sigma2 - rho * z1) * sqrt(inv_one_minus_rho_sq)
        total += -logdet_term - 0.5 * (z1^2 + z2^2)
        psi = alpha + uv1[i]
        theta = beta + uv2[i]
        lp_observed_zero = -log1pexp(-psi) +
            logdensityof(WD.BinomialLogit(K, theta), 0)
        lp_unobserved = logaddexp(lp_observed_zero, -log1pexp(psi))
        if i <= n
            total += log(omega)
            for j in 1:J
                count = X[i, j]
                if count > 0
                    total += -log1pexp(-psi) +
                        logdensityof(WD.BinomialLogit(K, theta), count)
                else
                    total += lp_unobserved
                end
            end
        else
            total += logaddexp(log1p(-omega), log(omega) + J * lp_unobserved)
        end
    end
    return total
end

@wren function multi_occupancy(X, K, J, n, S)
    alpha ~ WD.Cauchy(0.0, 2.5)
    beta ~ WD.Cauchy(0.0, 2.5)
    raw_omega ~ WD.Flat()
    raw_rho ~ WD.Flat()
    raw_sigma1 ~ WD.Flat()
    raw_sigma2 ~ WD.Flat()
    @iid uv1 ~ WD.Flat() size=S
    @iid uv2 ~ WD.Flat() size=S

    omega = inv(1 + exp(-raw_omega))
    rho_unit = inv(1 + exp(-raw_rho))
    rho = 2 * rho_unit - 1
    sigma1 = exp(raw_sigma1)
    sigma2 = exp(raw_sigma2)
    # The beta prior is on (rho + 1)/2; its factor has no extra transform
    # Jacobian. The raw rho coordinate does carry Stan's [-1,1] Jacobian.
    @addloglikelihood logdensityof(WD.Beta(2.0, 2.0), omega) +
        log(omega) + log1p(-omega) +
        logdensityof(WD.Beta(2.0, 2.0), rho_unit) +
        log(2.0) + log(rho_unit) + log1p(-rho_unit) +
        logdensityof(WD.Cauchy(0.0, 2.5), sigma1) + raw_sigma1 +
        logdensityof(WD.Cauchy(0.0, 2.5), sigma2) + raw_sigma2
    @addloglikelihood occupancy_logdensity(X, K, J, n, S, alpha, beta,
        omega, rho, sigma1, sigma2, uv1, uv2)
    return nothing
end

const wren_model = let
    data["n"] == 28 && data["J"] == 20 && data["K"] == 18 &&
        data["S"] == 50 || error("unexpected public butterfly fixture")
    multi_occupancy(data["X"], data["K"], data["J"], data["n"], data["S"])
end

end
