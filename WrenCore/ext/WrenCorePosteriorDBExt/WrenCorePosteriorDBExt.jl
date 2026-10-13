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
include("models/mh.jl")
include("models/rate1.jl")
include("models/radon_pooled_all.jl")
include("models/radon_pooled_mn.jl")
include("models/sblri_blr.jl")

# Maps PosteriorDB posterior names to the modules defining the models. Each module must
# define `wren_model`.
const MODELS = Dict(
    "arK-arK" => ArK,
    "eight_schools-eight_schools_centered" => EightSchoolsCentered,
    "eight_schools-eight_schools_noncentered" => EightSchoolsNoncentered,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson,
    "gp_pois_regr-gp_regr" => GPRegr,
    "kidiq-kidscore_interaction" => KidscoreInteraction,
    "Mh_data-Mh_model" => Mh,
    "Rate_1_data-Rate_1_model" => Rate1,
    "radon_all-radon_pooled" => RadonPooledAll,
    "radon_mn-radon_pooled" => RadonPooledMN,
    "sblri-blr" => SblriBLR,
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
