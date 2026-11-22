# PosteriorDB: the seven earnings-height regressions. They share a dataset and differ only
# in the response and predictors, so each module just builds its own `y` and `X`.
module EarningsBase

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
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

end

module EarnHeight
using ..EarningsBase: data, earnings_base
const wren_model = earnings_base(data["earn"], hcat(ones(data["N"]), data["height"]))
end

module Log10EarnHeight
using ..EarningsBase: data, earnings_base
const wren_model =
    earnings_base(log10.(data["earn"]), hcat(ones(data["N"]), data["height"]))
end

module LogEarnHeight
using ..EarningsBase: data, earnings_base
const wren_model = earnings_base(log.(data["earn"]), hcat(ones(data["N"]), data["height"]))
end

module LogEarnHeightMale
using ..EarningsBase: data, earnings_base
const wren_model =
    earnings_base(log.(data["earn"]), hcat(ones(data["N"]), data["height"], data["male"]))
end

module LogEarnInteraction
using ..EarningsBase: data, earnings_base
const wren_model = let
    height, male = data["height"], data["male"]
    earnings_base(log.(data["earn"]), hcat(ones(data["N"]), height, male, height .* male))
end
end

module LogEarnInteractionZ
using Statistics: mean, std
using ..EarningsBase: data, earnings_base
const wren_model = let
    height, male = data["height"], data["male"]
    z_height = (height .- mean(height)) ./ std(height)
    earnings_base(
        log.(data["earn"]),
        hcat(ones(data["N"]), z_height, male, z_height .* male),
    )
end
end

module LogEarnLogheightMale
using ..EarningsBase: data, earnings_base
const wren_model = earnings_base(
    log.(data["earn"]),
    hcat(ones(data["N"]), log.(data["height"]), data["male"]),
)
end
