# PosteriorDB: six Mesquite mass regressions. They share a dataset and differ only in the
# response and predictors, so each module just builds its own `y` and `X`.
module MesquiteBase

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

# Derived predictors used by the `logmesquite_logv*` models.
log_volume() = log.(data["diam1"] .* data["diam2"] .* data["canopy_height"])
log_area() = log.(data["diam1"] .* data["diam2"])
log_shape() = log.(data["diam1"] ./ data["diam2"])

end

module MesquiteRaw
using ..MesquiteBase: data, mesquite_base
const wren_model = mesquite_base(
    data["weight"],
    hcat(
        ones(data["N"]),
        data["diam1"],
        data["diam2"],
        data["canopy_height"],
        data["total_height"],
        data["density"],
        data["group"],
    ),
)
end

module LogMesquite
using ..MesquiteBase: data, mesquite_base
const wren_model = mesquite_base(
    log.(data["weight"]),
    hcat(
        ones(data["N"]),
        log.(data["diam1"]),
        log.(data["diam2"]),
        log.(data["canopy_height"]),
        log.(data["total_height"]),
        log.(data["density"]),
        data["group"],
    ),
)
end

module LogMesquiteLogVA
using ..MesquiteBase: data, mesquite_base, log_volume, log_area
const wren_model = mesquite_base(
    log.(data["weight"]),
    hcat(ones(data["N"]), log_volume(), log_area(), data["group"]),
)
end

module LogMesquiteLogVAS
using ..MesquiteBase: data, mesquite_base, log_volume, log_area, log_shape
const wren_model = mesquite_base(
    log.(data["weight"]),
    hcat(
        ones(data["N"]),
        log_volume(),
        log_area(),
        log_shape(),
        log.(data["total_height"]),
        log.(data["density"]),
        data["group"],
    ),
)
end

module LogMesquiteLogVASH
using ..MesquiteBase: data, mesquite_base, log_volume, log_area, log_shape
const wren_model = mesquite_base(
    log.(data["weight"]),
    hcat(
        ones(data["N"]),
        log_volume(),
        log_area(),
        log_shape(),
        log.(data["total_height"]),
        data["group"],
    ),
)
end

module LogMesquiteLogVolume
using ..MesquiteBase: data, mesquite_base, log_volume
const wren_model = mesquite_base(log.(data["weight"]), hcat(ones(data["N"]), log_volume()))
end
