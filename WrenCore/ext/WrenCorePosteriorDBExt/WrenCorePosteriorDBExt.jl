module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/ark.jl")
include("models/eight_schools_centered.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/kidscore_interaction.jl")
include("models/mh.jl")
include("models/rate1.jl")
include("models/sblri_blr.jl")
include("models/wells_dist.jl")
include("models/wells_dist100.jl")
include("models/wells_interaction.jl")
include("models/wells_interaction_c.jl")

# Maps PosteriorDB posterior names to the modules defining the models. Each module must
# define `wren_model`.
const MODELS = Dict(
    "arK-arK" => ArK,
    "eight_schools-eight_schools_centered" => EightSchoolsCentered,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson,
    "gp_pois_regr-gp_regr" => GPRegr,
    "kidiq-kidscore_interaction" => KidscoreInteraction,
    "Mh_data-Mh_model" => Mh,
    "Rate_1_data-Rate_1_model" => Rate1,
    "sblri-blr" => SblriBLR,
    "wells_data-wells_dist" => WellsDist,
    "wells_data-wells_dist100_model" => WellsDist100,
    "wells_data-wells_interaction_model" => WellsInteraction,
    "wells_data-wells_interaction_c_model" => WellsInteractionC,
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
