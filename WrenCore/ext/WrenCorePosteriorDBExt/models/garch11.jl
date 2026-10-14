# PosteriorDB: garch-garch11
module Garch11

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "garch-garch11")))
end

@wren function garch11(y, sigma1)
    mu ~ WD.Flat()
    log_alpha0 ~ WD.Flat()
    raw_alpha1 ~ WD.Flat()
    raw_beta1 ~ WD.Flat()
    alpha0 = exp(log_alpha0)
    alpha1 = inv(1 + exp(-raw_alpha1))
    beta_ratio = inv(1 + exp(-raw_beta1))
    beta1 = (1 - alpha1) * beta_ratio
    # Stan has flat density on alpha0 > 0, alpha1 in (0, 1), and
    # beta1 in (0, 1-alpha1). The joint transform is triangular.
    # Positive-flat alpha0 needs #192; the beta1 bound depending on alpha1
    # needs #207. Both are temporary manual-target transforms.
    @addloglikelihood log_alpha0 + log(alpha1) + 2 * log1p(-alpha1) +
        log(beta_ratio) + log1p(-beta_ratio)
    sigma = Vector{typeof(alpha0 + mu)}(undef, length(y))
    sigma[1] = sigma1
    for t in 2:length(y)
        sigma[t] = sqrt(alpha0 + alpha1 * (y[t - 1] - mu)^2 +
            beta1 * sigma[t - 1]^2)
    end
    @batch for t in eachindex(y)
        @observe y[t] ~ WD.Normal(mu, sigma[t])
    end
    return nothing
end

const wren_model = garch11(data["y"], data["sigma1"])

end
