# PosteriorDB: the six Mesquite mass regressions. They share a dataset and differ only in
# the response and predictors.
module Mesquite

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "mesquite-mesquite")))
end

@wren function mesquite_base(y, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    sigma ~ WD.PositiveFlat()
    mu = X * beta
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const weight = data["weight"]
const diam1 = data["diam1"]
const diam2 = data["diam2"]
const canopy_height = data["canopy_height"]
const total_height = data["total_height"]
const density = data["density"]
const group = data["group"]
const intercept = ones(data["N"])
# Derived predictors used by the `logmesquite_logv*` models.
const log_volume = log.(diam1 .* diam2 .* canopy_height)
const log_area = log.(diam1 .* diam2)
const log_shape = log.(diam1 ./ diam2)

# mesquite-mesquite
const mesquite = mesquite_base(
    weight,
    hcat(intercept, diam1, diam2, canopy_height, total_height, density, group),
)
# mesquite-logmesquite
const logmesquite = mesquite_base(
    log.(weight),
    hcat(
        intercept,
        log.(diam1),
        log.(diam2),
        log.(canopy_height),
        log.(total_height),
        log.(density),
        group,
    ),
)
# mesquite-logmesquite_logva
const logmesquite_logva =
    mesquite_base(log.(weight), hcat(intercept, log_volume, log_area, group))
# mesquite-logmesquite_logvas
const logmesquite_logvas = mesquite_base(
    log.(weight),
    hcat(
        intercept,
        log_volume,
        log_area,
        log_shape,
        log.(total_height),
        log.(density),
        group,
    ),
)
# mesquite-logmesquite_logvash
const logmesquite_logvash = mesquite_base(
    log.(weight),
    hcat(intercept, log_volume, log_area, log_shape, log.(total_height), group),
)
# mesquite-logmesquite_logvolume
const logmesquite_logvolume = mesquite_base(log.(weight), hcat(intercept, log_volume))

end
