# PosteriorDB: fims_Aus_Jpn_irt-2pl_latent_reg_irt
module FIMS2PL

using PosteriorDB
using Statistics: mean, std
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
    )))
end

function adjusted_covariates(W)
    J, K = size(W)
    W_adj = Matrix{Float64}(undef, J, K)
    W_adj[:, 1] .= W[:, 1]
    for k in 2:K
        column = view(W, :, k)
        lo, hi = extrema(column)
        scale = all(x -> x == lo || x == hi, column) ? hi - lo : 2 * std(column)
        center = mean(column)
        W_adj[:, k] .= (column .- center) ./ scale
    end
    return W_adj
end

function fims_logdensity(y, ii, jj, W_adj, alpha, beta_free, theta, lambda_adj)
    I, J = length(alpha), length(theta)
    total = zero(alpha[1] + beta_free[1] + theta[1] + lambda_adj[1])
    beta_last = -sum(beta_free)
    for i in eachindex(beta_free)
        total += logdensityof(WD.Normal(0.0, 3.0), beta_free[i])
    end
    total += logdensityof(WD.Normal(0.0, 3.0), beta_last)
    for j in 1:J
        location = zero(total)
        for k in eachindex(lambda_adj)
            location += W_adj[j, k] * lambda_adj[k]
        end
        total += logdensityof(WD.Normal(location, 1.0), theta[j])
    end
    for n in eachindex(y)
        i = ii[n]
        beta_i = i == I ? beta_last : beta_free[i]
        total += logdensityof(WD.BernoulliLogit(alpha[i] * theta[jj[n]] - beta_i), y[n])
    end
    return total
end

@wren function latent_reg_2pl(y, ii, jj, W_adj, I, J, K)
    @iid alpha ~ WD.LogNormal(1.0, 1.0) size=I
    @iid beta_free ~ WD.Flat() size=(I - 1)
    @iid theta ~ WD.Flat() size=J
    @iid lambda_adj ~ WD.TDist(3.0, 0.0, 1.0) size=K
    @addloglikelihood fims_logdensity(y, ii, jj, W_adj,
        alpha, beta_free, theta, lambda_adj)
    return nothing
end

const wren_model = let
    data["I"] == 14 && data["J"] == 500 && data["N"] == 7000 &&
        data["K"] == 4 || error("unexpected public FIMS fixture")
    W_adj = adjusted_covariates(data["W"])
    latent_reg_2pl(data["y"], data["ii"], data["jj"], W_adj,
        data["I"], data["J"], data["K"])
end

end
