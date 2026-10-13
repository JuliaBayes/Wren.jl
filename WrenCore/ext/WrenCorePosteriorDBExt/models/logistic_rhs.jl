# PosteriorDB: ovarian/prostate-logistic_regression_rhs
module LogisticRHS

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

function rhs_logdensity(y, x, scale_global, nu_global, nu_local,
        slab_scale, slab_df, beta0, z, raw_tau, raw_lambda, raw_caux)
    tau = exp(raw_tau)
    caux = exp(raw_caux)
    shape = 0.5 * slab_df
    total = logdensityof(WD.TDist(nu_global, 0.0, 2 * scale_global), tau) +
        raw_tau
    # InverseGamma(shape, shape) on caux is Gamma(shape, 1/shape) on 1/caux.
    # The precision transform and caux = exp(raw_caux) leave -raw_caux.
    total += logdensityof(WD.Gamma(shape, inv(shape)), inv(caux)) - raw_caux
    c2 = slab_scale^2 * caux
    for raw in raw_lambda
        total += logdensityof(WD.TDist(nu_local, 0.0, 1.0), exp(raw)) + raw
    end
    beta = [begin
        lambda_j = exp(raw_lambda[j])
        z[j] * tau * sqrt(c2 * lambda_j^2 /
            (c2 + tau^2 * lambda_j^2))
    end for j in eachindex(z)]
    for i in eachindex(y)
        eta = beta0
        for j in eachindex(beta)
            eta += x[i, j] * beta[j]
        end
        total += logdensityof(WD.BernoulliLogit(eta), y[i])
    end
    return total
end

@wren function logistic_rhs(y, x, n, d, scale_icept, scale_global,
        nu_global, nu_local, slab_scale, slab_df)
    beta0 ~ WD.Normal(0.0, scale_icept)
    @iid z ~ WD.Normal(0.0, 1.0) size=d
    raw_tau ~ WD.Flat()
    @iid raw_lambda ~ WD.Flat() size=d
    raw_caux ~ WD.Flat()
    @addloglikelihood rhs_logdensity(y, x, scale_global, nu_global,
        nu_local, slab_scale, slab_df, beta0, z, raw_tau, raw_lambda,
        raw_caux)
    return nothing
end

function load_case(name, n, d)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    data["n"] == n && data["d"] == d ||
        error("unexpected public regularized horseshoe fixture: $name")
    return logistic_rhs(data["y"], Float64.(data["x"]), n, d,
        data["scale_icept"], data["scale_global"],
        data["nu_global"], data["nu_local"],
        data["slab_scale"], data["slab_df"])
end

const ovarian = (wren_model=load_case("ovarian-logistic_regression_rhs", 54, 1536),)
const prostate = (wren_model=load_case("prostate-logistic_regression_rhs", 102, 5966),)

end
