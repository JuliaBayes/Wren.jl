# PosteriorDB: soil_carbon-soil_incubation
module SoilIncubation

using PosteriorDB
using WrenCore: @wren, @observe, @batch
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

@wren function soil_incubation(ts, observed, total, t0)
    k1 ~ WD.HalfNormal(0.0, 1.0)
    k2 ~ WD.HalfNormal(0.0, 1.0)
    alpha21 ~ WD.HalfNormal(0.0, 1.0)
    alpha12 ~ WD.HalfNormal(0.0, 1.0)
    gamma ~ WD.Beta(10.0, 1.0)
    sigma ~ WD.HalfCauchy(0.0, 1.0)
    mu = Vector{typeof(k1 + k2 + alpha21 + alpha12 + gamma + sigma)}(
        undef, length(ts))
    for i in eachindex(ts)
        mu[i] = evolved_carbon(ts[i] - t0, total, gamma, k1, k2,
            alpha21, alpha12)
    end
    @batch for i in eachindex(ts)
        @observe observed[i] ~ WD.Normal(mu[i], sigma)
    end
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
