# PosteriorDB: loss_curves-losscurve_sislob
module LossCurveSislob

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "loss_curves-losscurve_sislob",
    )))
end

function losscurve_logp(loss, cohort_id, t_idx, t_value, premium,
                        growthmodel_id, omega, theta, LR, loss_sd)
    total = zero(omega + theta + LR[1] + loss_sd)
    for i in eachindex(loss)
        t = t_value[t_idx[i]]
        growth = if growthmodel_id == 1
            1 - exp(-(t / theta)^omega)
        else
            t^omega / (t^omega + theta^omega)
        end
        cohort = cohort_id[i]
        mean_loss = LR[cohort] * premium[cohort] * growth
        total += logdensityof(WD.Normal(mean_loss, loss_sd * premium[cohort]), loss[i])
    end
    return total
end

@wren function losscurve_sislob(loss, cohort_id, t_idx, t_value, premium,
                                growthmodel_id, n_cohort)
    omega ~ WD.LogNormal(0.0, 0.5)
    theta ~ WD.LogNormal(0.0, 0.5)
    # Wren evaluates these before LR, whose conditional prior needs their values.
    mu_LR ~ WD.Normal(0.0, 0.5)
    sd_LR ~ WD.LogNormal(0.0, 0.5)
    @iid LR ~ WD.LogNormal(mu_LR, sd_LR) size=n_cohort
    loss_sd ~ WD.LogNormal(0.0, 0.7)
    @addloglikelihood losscurve_logp(loss, cohort_id, t_idx, t_value, premium,
                                     growthmodel_id, omega, theta, LR, loss_sd)
    return nothing
end

const wren_model = losscurve_sislob(
    data["loss"], data["cohort_id"], data["t_idx"], data["t_value"],
    data["premium"], data["growthmodel_id"], data["n_cohort"],
)

end
