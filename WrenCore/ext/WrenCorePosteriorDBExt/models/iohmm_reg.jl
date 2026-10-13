# PosteriorDB: iohmm_reg_simulated-iohmm_reg
module IOHMMReg

using PosteriorDB
using LogExpFunctions: logaddexp
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..HMMGaussian: simplex3

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "iohmm_reg_simulated-iohmm_reg",
    )))
end

linear_predictor(u, t, w) =
    u[t, 1] * w[1] + u[t, 2] * w[2] + u[t, 3] * w[3] + u[t, 4] * w[4]

@wren function iohmm_reg(y, u)
    raw_pi1 ~ WD.Flat()
    raw_pi2 ~ WD.Flat()
    @iid w1 ~ WD.Normal(0.0, 5.0) size=4
    @iid w2 ~ WD.Normal(0.0, 5.0) size=4
    @iid w3 ~ WD.Normal(0.0, 5.0) size=4
    @iid b1 ~ WD.Normal(0.0, 5.0) size=4
    @iid b2 ~ WD.Normal(0.0, 5.0) size=4
    @iid b3 ~ WD.Normal(0.0, 5.0) size=4
    raw_sigma1 ~ WD.Flat()
    raw_sigma2 ~ WD.Flat()
    raw_sigma3 ~ WD.Flat()

    pi, pi_jac = simplex3(raw_pi1, raw_pi2)
    sigma1 = exp(raw_sigma1)
    sigma2 = exp(raw_sigma2)
    sigma3 = exp(raw_sigma3)
    # Stan constrains the three sigmas but uses untruncated Normal prior factors.
    @addloglikelihood pi_jac + raw_sigma1 + raw_sigma2 + raw_sigma3 +
        logdensityof(WD.Normal(0.0, 3.0), sigma1) +
        logdensityof(WD.Normal(0.0, 3.0), sigma2) +
        logdensityof(WD.Normal(0.0, 3.0), sigma3)

    g1 = log(pi[1]) +
        logdensityof(WD.Normal(linear_predictor(u, 1, b1), sigma1), y[1])
    g2 = log(pi[2]) +
        logdensityof(WD.Normal(linear_predictor(u, 1, b2), sigma2), y[1])
    g3 = log(pi[3]) +
        logdensityof(WD.Normal(linear_predictor(u, 1, b3), sigma3), y[1])
    for t in 2:length(y)
        r1 = linear_predictor(u, t, w1)
        r2 = linear_predictor(u, t, w2)
        r3 = linear_predictor(u, t, w3)
        normalizer = logaddexp(logaddexp(r1, r2), r3)
        # The source Stan program uses logA[t][i] for every destination j.
        # Thus this transition factor is shared across all three next states.
        shared = logaddexp(logaddexp(g1 + r1 - normalizer,
            g2 + r2 - normalizer), g3 + r3 - normalizer)
        g1 = shared +
            logdensityof(WD.Normal(linear_predictor(u, t, b1), sigma1), y[t])
        g2 = shared +
            logdensityof(WD.Normal(linear_predictor(u, t, b2), sigma2), y[t])
        g3 = shared +
            logdensityof(WD.Normal(linear_predictor(u, t, b3), sigma3), y[t])
    end
    @addloglikelihood logaddexp(logaddexp(g1, g2), g3)
    return nothing
end

const wren_model = let
    data["K"] == 3 && data["M"] == 4 ||
        error("public input-output HMM fixture requires K=3, M=4")
    iohmm_reg(data["y"], data["u"])
end

end
