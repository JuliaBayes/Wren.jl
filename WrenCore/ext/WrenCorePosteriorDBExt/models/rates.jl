# PosteriorDB: Rate_1 to Rate_5.
module Rates

using PosteriorDB
using WrenCore: @wren, @observe
using WrenDists: WrenDists as WD

load_data(name) = PosteriorDB.load(
    PosteriorDB.dataset(PosteriorDB.posterior(PosteriorDB.database(), name)),
)

@wren function rate_1_model(k, n)
    theta ~ WD.Uniform(0.0, 1.0)
    @observe k ~ WD.Binomial(n, theta)
    return nothing
end

@wren function rate_2_model(k1, n1, k2, n2)
    theta1 ~ WD.Beta(1.0, 1.0)
    theta2 ~ WD.Beta(1.0, 1.0)
    @observe k1 ~ WD.Binomial(n1, theta1)
    @observe k2 ~ WD.Binomial(n2, theta2)
    return nothing
end

@wren function rate_3_model(k1, n1, k2, n2)
    theta ~ WD.Beta(1.0, 1.0)
    @observe k1 ~ WD.Binomial(n1, theta)
    @observe k2 ~ WD.Binomial(n2, theta)
    return nothing
end

@wren function rate_4_model(k, n)
    theta ~ WD.Beta(1.0, 1.0)
    thetaprior ~ WD.Beta(1.0, 1.0)
    @observe k ~ WD.Binomial(n, theta)
    return nothing
end

const rate1 = let
    data = load_data("Rate_1_data-Rate_1_model")
    rate_1_model(data["k"], data["n"])
end
const rate2 = let
    data = load_data("Rate_2_data-Rate_2_model")
    rate_2_model(data["k1"], data["n1"], data["k2"], data["n2"])
end
const rate3 = let
    data = load_data("Rate_3_data-Rate_3_model")
    rate_3_model(data["k1"], data["n1"], data["k2"], data["n2"])
end
const rate4 = let
    data = load_data("Rate_4_data-Rate_4_model")
    rate_4_model(data["k"], data["n"])
end
# The generated predictive counts in the Stan source are outside the target density, so
# Rate_5 has the same density as Rate_3.
const rate5 = let
    data = load_data("Rate_5_data-Rate_5_model")
    rate_3_model(data["k1"], data["n1"], data["k2"], data["n2"])
end

end
