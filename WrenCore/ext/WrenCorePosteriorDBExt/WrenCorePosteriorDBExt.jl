module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/ark.jl")
include("models/earnings.jl")
include("models/eight_schools.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/kidscore.jl")
include("models/kilpisjarvi.jl")
include("models/mesquite.jl")
include("models/mh.jl")
include("models/nes.jl")
include("models/pilots.jl")
include("models/radon_pooled.jl")
include("models/rates.jl")
include("models/rats_model.jl")
include("models/sblri_blr.jl")
include("models/wells.jl")

# Maps PosteriorDB posterior names to the Wren models.
const MODELS = Dict(
    "arK-arK" => ArK.model,
    "earnings-earn_height" => Earnings.earn_height,
    "earnings-log10earn_height" => Earnings.log10earn_height,
    "earnings-logearn_height" => Earnings.logearn_height,
    "earnings-logearn_height_male" => Earnings.logearn_height_male,
    "earnings-logearn_interaction" => Earnings.logearn_interaction,
    "earnings-logearn_interaction_z" => Earnings.logearn_interaction_z,
    "earnings-logearn_logheight_male" => Earnings.logearn_logheight_male,
    "eight_schools-eight_schools_centered" => EightSchools.centered,
    "eight_schools-eight_schools_noncentered" => EightSchools.noncentered,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial.model,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson.model,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson.model,
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
    "pilots-pilots" => Pilots.model,
    "radon_all-radon_pooled" => RadonPooled.radon_all,
    "radon_mn-radon_pooled" => RadonPooled.radon_mn,
    "Rate_1_data-Rate_1_model" => Rates.rate1,
    "Rate_2_data-Rate_2_model" => Rates.rate2,
    "Rate_3_data-Rate_3_model" => Rates.rate3,
    "Rate_4_data-Rate_4_model" => Rates.rate4,
    "Rate_5_data-Rate_5_model" => Rates.rate5,
    "rats_data-rats_model" => RatsModel.model,
    "sblri-blr" => SblriBLR.model,
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
