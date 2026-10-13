# PosteriorDB: gp_pois_regr-gp_pois_regr
module GPPoissonRegression

using PosteriorDB
using LinearAlgebra: Symmetric, I, cholesky, issuccess
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "gp_pois_regr-gp_pois_regr")))
end

function gp_pois_logp(counts, dist_sq, rho, alpha, f_tilde)
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + 1e-10I
    chol = cholesky(Symmetric(covariance); check=false)
    issuccess(chol) || return -Inf
    f = chol.L * f_tilde
    total = zero(f[1])
    for i in eachindex(counts)
        total += logdensityof(WD.PoissonLog(f[i]), counts[i])
    end
    return total
end

@wren function gp_pois_regr(counts, dist_sq, N)
    rho ~ WD.Gamma(25.0, 1 / 4)
    # Stan's positive Normal factor differs by the constant log(2).
    alpha ~ WD.HalfNormal(0.0, 2.0)
    @iid f_tilde ~ WD.Normal(0.0, 1.0) size=N
    @addloglikelihood gp_pois_logp(counts, dist_sq, rho, alpha, f_tilde)
    return nothing
end

const wren_model = let
    x = data["x"]
    gp_pois_regr(data["k"], (x .- x') .^ 2, data["N"])
end

end
