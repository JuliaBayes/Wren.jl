module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/accel_splines.jl")
include("models/ark.jl")
include("models/arma11.jl")
include("models/covid_imperial.jl")
include("models/diamonds.jl")
include("models/dugongs.jl")
include("models/earnings.jl")
include("models/eight_schools.jl")
include("models/garch11.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_pois_regr.jl")
include("models/gp_regr.jl")
include("models/kidscore.jl")
include("models/kilpisjarvi.jl")
include("models/logaddexp.jl")
include("models/logistic_rhs.jl")
include("models/losscurve_sislob.jl")
include("models/low_dim_gauss_mix_collapse.jl")
include("models/low_dim_gauss_mix.jl")
include("models/mcycle_gp.jl")
include("models/mesquite.jl")
include("models/mh.jl")
include("models/nes.jl")
include("models/normal_mixture.jl")
include("models/normal_mixture_k.jl")
include("models/pilots.jl")
include("models/presidential_gp.jl")
include("models/prophet.jl")
include("models/radon_pooled.jl")
include("models/rates.jl")
include("models/rats_model.jl")
include("models/sblri_blr.jl")
include("models/sblrc_blr.jl")
include("models/seeds.jl")
include("models/seeds_centered.jl")
include("models/seeds_stanified.jl")
include("models/sesame_one_pred.jl")
include("models/soil_incubation.jl")
include("models/surgical_model.jl")
include("models/uk_drivers.jl")
include("models/wells.jl")

# Maps PosteriorDB posterior names to the Wren models.
const MODELS = Dict(
    "arK-arK" => ArK.model,
    "arma-arma11" => ARMA11.model,
    "diamonds-diamonds" => Diamonds.model,
    "dugongs_data-dugongs_model" => Dugongs.model,
    "earnings-earn_height" => Earnings.earn_height,
    "earnings-log10earn_height" => Earnings.log10earn_height,
    "earnings-logearn_height" => Earnings.logearn_height,
    "earnings-logearn_height_male" => Earnings.logearn_height_male,
    "earnings-logearn_interaction" => Earnings.logearn_interaction,
    "earnings-logearn_interaction_z" => Earnings.logearn_interaction_z,
    "earnings-logearn_logheight_male" => Earnings.logearn_logheight_male,
    "ecdc0401-covid19imperial_v2" => CovidImperial.ecdc0401_v2,
    "ecdc0401-covid19imperial_v3" => CovidImperial.ecdc0401_v3,
    "ecdc0501-covid19imperial_v2" => CovidImperial.ecdc0501_v2,
    "ecdc0501-covid19imperial_v3" => CovidImperial.ecdc0501_v3,
    "eight_schools-eight_schools_centered" => EightSchools.centered,
    "eight_schools-eight_schools_noncentered" => EightSchools.noncentered,
    "garch-garch11" => Garch11.model,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial.model,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson.model,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson.model,
    "gp_pois_regr-gp_pois_regr" => GPPoissonRegression.model,
    "gp_pois_regr-gp_regr" => GPRegr.model,
    "kidiq-kidscore_interaction" => Kidscore.interaction,
    "kidiq-kidscore_momhs" => Kidscore.momhs,
    "kidiq-kidscore_momhsiq" => Kidscore.momhsiq,
    "kidiq-kidscore_momiq" => Kidscore.momiq,
    "kidiq_with_mom_work-kidscore_interaction_c" => Kidscore.interaction_c,
    "kidiq_with_mom_work-kidscore_interaction_c2" => Kidscore.interaction_c2,
    "kidiq_with_mom_work-kidscore_interaction_z" => Kidscore.interaction_z,
    "kidiq_with_mom_work-kidscore_mom_work" => Kidscore.mom_work,
    "kilpisjarvi_mod-kilpisjarvi" => Kilpisjarvi.model,
    "loss_curves-losscurve_sislob" => LossCurveSislob.model,
    "low_dim_gauss_mix-low_dim_gauss_mix" => LowDimGaussMix.model,
    "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse" =>
        LowDimGaussMixCollapse.model,
    "mcycle_gp-accel_gp" => McycleGP.model,
    "mcycle_splines-accel_splines" => AccelSplines.model,
    "mesquite-logmesquite" => Mesquite.logmesquite,
    "mesquite-logmesquite_logva" => Mesquite.logmesquite_logva,
    "mesquite-logmesquite_logvas" => Mesquite.logmesquite_logvas,
    "mesquite-logmesquite_logvash" => Mesquite.logmesquite_logvash,
    "mesquite-logmesquite_logvolume" => Mesquite.logmesquite_logvolume,
    "mesquite-mesquite" => Mesquite.mesquite,
    "Mh_data-Mh_model" => Mh.model,
    "nes1972-nes" => NES.nes1972,
    "nes1976-nes" => NES.nes1976,
    "nes1980-nes" => NES.nes1980,
    "nes1984-nes" => NES.nes1984,
    "nes1988-nes" => NES.nes1988,
    "nes1992-nes" => NES.nes1992,
    "nes1996-nes" => NES.nes1996,
    "nes2000-nes" => NES.nes2000,
    "normal_2-normal_mixture" => NormalMixture.model,
    "normal_5-normal_mixture_k" => NormalMixtureK.model,
    "ovarian-logistic_regression_rhs" => LogisticRHS.ovarian,
    "pilots-pilots" => Pilots.model,
    "prostate-logistic_regression_rhs" => LogisticRHS.prostate,
    "radon_all-radon_pooled" => RadonPooled.radon_all,
    "radon_mn-radon_pooled" => RadonPooled.radon_mn,
    "Rate_1_data-Rate_1_model" => Rates.rate1,
    "Rate_2_data-Rate_2_model" => Rates.rate2,
    "Rate_3_data-Rate_3_model" => Rates.rate3,
    "Rate_4_data-Rate_4_model" => Rates.rate4,
    "Rate_5_data-Rate_5_model" => Rates.rate5,
    "rats_data-rats_model" => RatsModel.model,
    "rstan_downloads-prophet" => Prophet.model,
    "sblrc-blr" => SblrcBLR.model,
    "sblri-blr" => SblriBLR.model,
    "seeds_data-seeds_centered_model" => SeedsCentered.model,
    "seeds_data-seeds_model" => Seeds.model,
    "seeds_data-seeds_stanified_model" => SeedsStanified.model,
    "sesame_data-sesame_one_pred_a" => SesameOnePred.model,
    "soil_carbon-soil_incubation" => SoilIncubation.model,
    "state_wide_presidential_votes-hierarchical_gp" => PresidentialGP.model,
    "surgical_data-surgical_model" => SurgicalModel.model,
    "uk_drivers-state_space_stochastic_level_stochastic_seasonal" => UKDrivers.model,
    "wells_data-wells_daae_c_model" => Wells.daae_c_model,
    "wells_data-wells_dae_c_model" => Wells.dae_c_model,
    "wells_data-wells_dae_inter_model" => Wells.dae_inter_model,
    "wells_data-wells_dae_model" => Wells.dae_model,
    "wells_data-wells_dist" => Wells.wells_dist,
    "wells_data-wells_dist100_model" => Wells.dist100_model,
    "wells_data-wells_dist100ars_model" => Wells.dist100ars_model,
    "wells_data-wells_interaction_c_model" => Wells.interaction_c_model,
    "wells_data-wells_interaction_model" => Wells.interaction_model,
)
const MODEL_NAMES = sort!(collect(keys(MODELS)))

"""
    posteriordb_model(name::AbstractString) -> WrenCore.Model

Return a Wren model for the PosteriorDB posterior called `name`, e.g. `"arK-arK"`, using the
data from PosteriorDB.

This requires PosteriorDB.jl and WrenDists.jl to be loaded. Only some posteriors are
available; an error is thrown for any others, which lists the available ones.
"""
function WrenCore.posteriordb_model(name::AbstractString)
    haskey(MODELS, name) || throw(
        ArgumentError(
            "no Wren model for the PosteriorDB posterior \"$name\". Available posteriors " *
            "are: " *
            join(MODEL_NAMES, ", "),
        ),
    )
    return MODELS[name]
end

"""
    available_posteriordb_models() -> Vector{String}

Return a vector of PosteriorDB model names which WrenCore has implementations for.
"""
WrenCore.available_posteriordb_models() = MODEL_NAMES

end # module
