# PosteriorDB: Mh_data-Mh_model
# Deviations: detected/undetected partitioning is done at model construction time
# rather than inside the model loop.
module Mh

using PosteriorDB
using LogExpFunctions: log1pexp, logaddexp, logit
using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Mh_data-Mh_model")))
end

@wren function mh_model(captures, detected, undetected, trials, M)
    omega ~ WD.Uniform(0.0, 1.0)
    mean_p ~ WD.Uniform(0.0, 1.0)
    sigma ~ WD.Uniform(0.0, 5.0)
    @iid eps_raw ~ WD.Normal(0.0, 1.0) size=M
    logit_p = logit(mean_p) .+ sigma .* eps_raw
    logit_p_detected = logit_p[detected]
    @batch for i in eachindex(captures)
        @observe captures[i] ~ WD.BinomialLogit(trials, logit_p_detected[i])
    end
    @addloglikelihood length(detected) * log(omega) + sum(
        logaddexp.(log(omega) .- trials .* log1pexp.(logit_p[undetected]), log1p(-omega)),
    )
    return nothing
end

const model = let
    M = data["M"]
    trials = data["T"]
    y = data["y"]
    detected = findall(>(0), y)
    undetected = findall(==(0), y)
    captures = y[detected]
    mh_model(captures, detected, undetected, trials, M)
end

end
