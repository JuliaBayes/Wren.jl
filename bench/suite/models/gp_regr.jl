# PosteriorDB: gp_pois_regr-gp_regr
module GPRegr

using PosteriorDB
using LinearAlgebra: Symmetric, I, cholesky
using Wren: Wren, @wren, @observe
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "gp_pois_regr-gp_regr")),
    )
end

@wren function gp_regr(y, dist_sq, N)
    rho ~ WD.Gamma(25.0, 1 / 4)
    alpha ~ WD.HalfNormal(0.0, 2.0)
    sigma ~ WD.HalfNormal(0.0, 1.0)
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + sigma * I
    @observe y ~ WD.MvNormalScale(zeros(N), cholesky(Symmetric(covariance)).L)
    return nothing
end

const wren_model = let
    x = data["x"]
    y = data["y"]
    gp_regr(y, (x .- x') .^ 2, data["N"])
end

end
