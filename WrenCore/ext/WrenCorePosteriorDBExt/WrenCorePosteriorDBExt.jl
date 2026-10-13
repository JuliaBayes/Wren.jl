module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/ark.jl")
include("models/eight_schools_centered.jl")
include("models/eight_schools_noncentered.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/kidscore_interaction.jl")
include("models/kidscore_momhs.jl")
include("models/kidscore_momhsiq.jl")
include("models/kidscore_momiq.jl")
include("models/mh.jl")
include("models/radon_pooled_all.jl")
include("models/radon_pooled_mn.jl")
include("models/rate1.jl")
include("models/rate2.jl")
include("models/rate3.jl")
include("models/rate4.jl")
include("models/rate5.jl")
include("models/sblri_blr.jl")
include("models/wells_dist.jl")
include("models/wells_dist100.jl")
include("models/wells_interaction.jl")
include("models/wells_interaction_c.jl")

# Maps PosteriorDB posterior names to the modules defining the models. Each module must
# define `wren_model`.
const MODELS = Dict(
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson,
    "Mh_data-Mh_model" => Mh,
    "Rate_1_data-Rate_1_model" => Rate1,
    "Rate_2_data-Rate_2_model" => Rate2,
    "Rate_3_data-Rate_3_model" => Rate3,
    "Rate_4_data-Rate_4_model" => Rate4,
    "Rate_5_data-Rate_5_model" => Rate5,
    "arK-arK" => ArK,
    "eight_schools-eight_schools_centered" => EightSchoolsCentered,
    "eight_schools-eight_schools_noncentered" => EightSchoolsNoncentered,
    "gp_pois_regr-gp_regr" => GPRegr,
    "kidiq-kidscore_interaction" => KidscoreInteraction,
    "kidiq-kidscore_momhs" => KidscoreMomhs,
    "kidiq-kidscore_momhsiq" => KidscoreMomhsiq,
    "kidiq-kidscore_momiq" => KidscoreMomiq,
    "radon_all-radon_pooled" => RadonPooledAll,
    "radon_mn-radon_pooled" => RadonPooledMN,
    "sblri-blr" => SblriBLR,
    "wells_data-wells_dist" => WellsDist,
    "wells_data-wells_dist100_model" => WellsDist100,
    "wells_data-wells_interaction_c_model" => WellsInteractionC,
    "wells_data-wells_interaction_model" => WellsInteraction,
)
const MODEL_NAMES = sort!(collect(keys(MODELS)))

function WrenCore.posteriordb_model(name::AbstractString)
    haskey(MODELS, name) || throw(
        ArgumentError(
            "no Wren model for the PosteriorDB posterior \"$name\". Available posteriors " *
            "are: " *
            join(MODEL_NAMES, ", "),
        ),
    )
    return MODELS[name].wren_model
end

WrenCore.available_posteriordb_models() = MODEL_NAMES

end # module
