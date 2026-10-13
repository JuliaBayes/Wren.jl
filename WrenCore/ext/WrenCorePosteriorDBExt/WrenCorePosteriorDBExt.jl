module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/ark.jl")
include("models/dugongs.jl")
include("models/eight_schools_centered.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/kidscore_interaction.jl")
include("models/mh.jl")
include("models/normal_mixture.jl")
include("models/rate1.jl")
include("models/sblri_blr.jl")
include("models/seeds.jl")
include("models/seeds_centered.jl")
include("models/seeds_stanified.jl")

# Maps PosteriorDB posterior names to the modules defining the models. Each module must
# define `wren_model`.
const MODELS = Dict(
    "arK-arK" => ArK,
    "dugongs_data-dugongs_model" => Dugongs,
    "eight_schools-eight_schools_centered" => EightSchoolsCentered,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson,
    "gp_pois_regr-gp_regr" => GPRegr,
    "kidiq-kidscore_interaction" => KidscoreInteraction,
    "Mh_data-Mh_model" => Mh,
    "normal_2-normal_mixture" => NormalMixture,
    "Rate_1_data-Rate_1_model" => Rate1,
    "sblri-blr" => SblriBLR,
    "seeds_data-seeds_model" => Seeds,
    "seeds_data-seeds_centered_model" => SeedsCentered,
    "seeds_data-seeds_stanified_model" => SeedsStanified,
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
