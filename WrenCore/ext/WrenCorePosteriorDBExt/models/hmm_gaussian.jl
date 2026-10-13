# PosteriorDB: hmm_gaussian_simulated-hmm_gaussian
module HMMGaussian

using PosteriorDB
using LogExpFunctions: logaddexp
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "hmm_gaussian_simulated-hmm_gaussian",
    )))
end

function simplex3(a, b)
    z1 = inv(1 + exp(-(a - log(2))))
    z2 = inv(1 + exp(-b))
    remainder = 1 - z1
    jacobian = log(z1) + log1p(-z1) + log(remainder) +
        log(z2) + log1p(-z2)
    return (z1, remainder * z2, remainder * (1 - z2)), jacobian
end

@wren function hmm_gaussian(y)
    raw_pi1 ~ WD.Flat()
    raw_pi2 ~ WD.Flat()
    raw_a11 ~ WD.Flat()
    raw_a12 ~ WD.Flat()
    raw_a21 ~ WD.Flat()
    raw_a22 ~ WD.Flat()
    raw_a31 ~ WD.Flat()
    raw_a32 ~ WD.Flat()
    raw_mu1 ~ WD.Flat()
    raw_gap2 ~ WD.Flat()
    raw_gap3 ~ WD.Flat()
    raw_sigma1 ~ WD.Flat()
    raw_sigma2 ~ WD.Flat()
    raw_sigma3 ~ WD.Flat()

    pi, pi_jac = simplex3(raw_pi1, raw_pi2)
    a1, a1_jac = simplex3(raw_a11, raw_a12)
    a2, a2_jac = simplex3(raw_a21, raw_a22)
    a3, a3_jac = simplex3(raw_a31, raw_a32)
    mu1 = raw_mu1
    mu2 = mu1 + exp(raw_gap2)
    mu3 = mu2 + exp(raw_gap3)
    sigma1 = exp(raw_sigma1)
    sigma2 = exp(raw_sigma2)
    sigma3 = exp(raw_sigma3)
    # Stan's simplex, ordered, and positive-support unconstraining Jacobians.
    @addloglikelihood pi_jac + a1_jac + a2_jac + a3_jac +
        raw_gap2 + raw_gap3 + raw_sigma1 + raw_sigma2 + raw_sigma3

    g1 = log(pi[1]) + logdensityof(WD.Normal(mu1, sigma1), y[1])
    g2 = log(pi[2]) + logdensityof(WD.Normal(mu2, sigma2), y[1])
    g3 = log(pi[3]) + logdensityof(WD.Normal(mu3, sigma3), y[1])
    for t in 2:length(y)
        next1 = logdensityof(WD.Normal(mu1, sigma1), y[t]) +
            logaddexp(logaddexp(g1 + log(a1[1]), g2 + log(a2[1])),
                g3 + log(a3[1]))
        next2 = logdensityof(WD.Normal(mu2, sigma2), y[t]) +
            logaddexp(logaddexp(g1 + log(a1[2]), g2 + log(a2[2])),
                g3 + log(a3[2]))
        next3 = logdensityof(WD.Normal(mu3, sigma3), y[t]) +
            logaddexp(logaddexp(g1 + log(a1[3]), g2 + log(a2[3])),
                g3 + log(a3[3]))
        g1, g2, g3 = next1, next2, next3
    end
    @addloglikelihood logaddexp(logaddexp(g1, g2), g3)
    return nothing
end

const wren_model = let
    data["K"] == 3 || error("public Gaussian HMM fixture requires K=3")
    hmm_gaussian(data["y"])
end

end
