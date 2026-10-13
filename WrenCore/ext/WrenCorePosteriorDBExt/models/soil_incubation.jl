# PosteriorDB: soil_carbon-soil_incubation
module SoilIncubation

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD

# The two-pool ODE has a constant 2×2 coefficient matrix. Its total carbon
# therefore has this closed-form trajectory; no numerical ODE tolerance enters
# the log density or its gradient.
function evolved_carbon(t, total, gamma, k1, k2, alpha21, alpha12)
    mean_rate = (k1 + k2) / 2
    gap = sqrt(((k1 - k2) / 2)^2 + alpha12 * alpha21 * k1 * k2)
    slope = -k1 * gamma - k2 * (1 - gamma) +
        alpha21 * k1 * gamma + alpha12 * k2 * (1 - gamma)
    x = gap * t
    sinhc = abs(x) < 1e-4 ? 1 + x^2 / 6 + x^4 / 120 : sinh(x) / x
    remaining = total * exp(-mean_rate * t) *
        (cosh(x) + (slope + mean_rate) * t * sinhc)
    return total - remaining
end

function soil_logdensity(ts, observed, total, t0, raw_k1, raw_k2,
        raw_alpha21, raw_alpha12, gamma, raw_sigma)
    k1, k2 = exp(raw_k1), exp(raw_k2)
    alpha21, alpha12 = exp(raw_alpha21), exp(raw_alpha12)
    sigma = exp(raw_sigma)
    density = logdensityof(WD.Normal(0.0, 1.0), k1) + raw_k1 +
        logdensityof(WD.Normal(0.0, 1.0), k2) + raw_k2 +
        logdensityof(WD.Normal(0.0, 1.0), alpha21) + raw_alpha21 +
        logdensityof(WD.Normal(0.0, 1.0), alpha12) + raw_alpha12 +
        logdensityof(WD.Cauchy(0.0, 1.0), sigma) + raw_sigma
    for i in eachindex(ts)
        mean = evolved_carbon(ts[i] - t0, total, gamma, k1, k2,
            alpha21, alpha12)
        density += logdensityof(WD.Normal(mean, sigma), observed[i])
    end
    return density
end

@wren function soil_incubation(ts, observed, total, t0)
    raw_k1 ~ WD.Flat()
    raw_k2 ~ WD.Flat()
    raw_alpha21 ~ WD.Flat()
    raw_alpha12 ~ WD.Flat()
    gamma ~ WD.Beta(10.0, 1.0)
    raw_sigma ~ WD.Flat()
    @addloglikelihood soil_logdensity(ts, observed, total, t0, raw_k1,
        raw_k2, raw_alpha21, raw_alpha12, gamma, raw_sigma)
    return nothing
end

const wren_model = let
    name = "soil_carbon-soil_incubation"
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        PosteriorDB.database(), name,
    )))
    data["N_t"] == 25 && length(data["ts"]) == 25 &&
        length(data["eCO2mean"]) == 25 ||
        error("unexpected public soil carbon fixture")
    soil_incubation(data["ts"], data["eCO2mean"], data["totalC_t0"], data["t0"])
end

end
