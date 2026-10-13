# PosteriorDB: traffic_accident_nyc-bym2_offset_only
module TrafficBYM2

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "traffic_accident_nyc-bym2_offset_only",
    )))
end

logistic(x) = inv(1 + exp(-x))
logit_jacobian(x) = -log1p(exp(-x)) - log1p(exp(x))

function traffic_logdensity(y, log_E, node1, node2, scaling_factor,
        beta0, sigma, rho, theta, phi)
    total = zero(beta0 + sigma + rho + theta[1] + phi[1])
    unstructured = sqrt(1 - rho)
    spatial = sqrt(rho / scaling_factor)
    for n in eachindex(y)
        eta = log_E[n] + beta0 + sigma *
            (unstructured * theta[n] + spatial * phi[n])
        total += logdensityof(WD.PoissonLog(eta), y[n])
    end
    for e in eachindex(node1)
        total -= 0.5 * (phi[node1[e]] - phi[node2[e]])^2
    end
    total += logdensityof(WD.Normal(0.0, 0.001 * length(phi)), sum(phi))
    return total
end

@wren function bym2(y, log_E, node1, node2, scaling_factor, N)
    beta0 ~ WD.Normal(0.0, 1.0)
    raw_sigma ~ WD.Flat()
    raw_rho ~ WD.Flat()
    @iid theta ~ WD.Normal(0.0, 1.0) size=N
    @iid phi ~ WD.Flat() size=N
    sigma = exp(raw_sigma)
    rho = logistic(raw_rho)
    # Stan's positive Normal factor has no truncation normalizer.
    @addloglikelihood logdensityof(WD.Normal(0.0, 1.0), sigma) + raw_sigma +
        logdensityof(WD.Beta(0.5, 0.5), rho) + logit_jacobian(raw_rho)
    @addloglikelihood traffic_logdensity(y, log_E, node1, node2,
        scaling_factor, beta0, sigma, rho, theta, phi)
    return nothing
end

const wren_model = let
    data["N"] == 1921 && data["N_edges"] == 5461 ||
        error("unexpected public NYC traffic fixture")
    bym2(data["y"], log.(data["E"]), data["node1"], data["node2"],
        data["scaling_factor"], data["N"])
end

end
