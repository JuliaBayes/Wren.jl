# PosteriorDB: the seven earnings-height regressions. They share a dataset and differ only
# in the response and predictors.
module Earnings

using PosteriorDB
using Statistics: mean, std
using WrenCore: @wren
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "earnings-earn_height")),
    )
end

@wren function earnings_base(y, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    sigma ~ WD.PositiveFlat()
    mu = X * beta
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const earn = data["earn"]
const height = data["height"]
const male = data["male"]
const intercept = ones(data["N"])
const z_height = (height .- mean(height)) ./ std(height)

# earnings-earn_height
const earn_height = earnings_base(earn, hcat(intercept, height))
# earnings-log10earn_height
const log10earn_height = earnings_base(log10.(earn), hcat(intercept, height))
# earnings-logearn_height
const logearn_height = earnings_base(log.(earn), hcat(intercept, height))
# earnings-logearn_height_male
const logearn_height_male = earnings_base(log.(earn), hcat(intercept, height, male))
# earnings-logearn_interaction
const logearn_interaction =
    earnings_base(log.(earn), hcat(intercept, height, male, height .* male))
# earnings-logearn_interaction_z
const logearn_interaction_z =
    earnings_base(log.(earn), hcat(intercept, z_height, male, z_height .* male))
# earnings-logearn_logheight_male
const logearn_logheight_male =
    earnings_base(log.(earn), hcat(intercept, log.(height), male))

end
