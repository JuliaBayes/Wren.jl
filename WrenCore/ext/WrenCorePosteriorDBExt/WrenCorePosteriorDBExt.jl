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
include("models/radon_hierarchical_intercept_centered_all.jl")
include("models/radon_hierarchical_intercept_centered_mn.jl")
include("models/radon_county.jl")
include("models/radon_partially_pooled_centered_all.jl")
include("models/radon_partially_pooled_centered_mn.jl")
include("models/radon_variable_intercept_centered_all.jl")
include("models/radon_variable_intercept_centered_mn.jl")
include("models/radon_variable_intercept_slope_centered_all.jl")
include("models/radon_variable_intercept_slope_centered_mn.jl")
include("models/radon_variable_slope_centered_all.jl")
include("models/radon_variable_slope_centered_mn.jl")
include("models/sblri_blr.jl")

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
    "radon_all-radon_hierarchical_intercept_centered" => RadonHierarchicalInterceptCenteredAll,
    "radon_all-radon_partially_pooled_centered" => RadonPartiallyPooledCenteredAll,
    "radon_all-radon_variable_intercept_centered" => RadonVariableInterceptCenteredAll,
    "radon_all-radon_variable_intercept_slope_centered" => RadonVariableInterceptSlopeCenteredAll,
    "radon_all-radon_variable_slope_centered" => RadonVariableSlopeCenteredAll,
    "radon_mn-radon_hierarchical_intercept_centered" => RadonHierarchicalInterceptCenteredMn,
    "radon_mn-radon_partially_pooled_centered" => RadonPartiallyPooledCenteredMn,
    "radon_mn-radon_variable_intercept_centered" => RadonVariableInterceptCenteredMn,
    "radon_mn-radon_variable_intercept_slope_centered" => RadonVariableInterceptSlopeCenteredMn,
    "radon_mn-radon_variable_slope_centered" => RadonVariableSlopeCenteredMn,
    "radon_mod-radon_county" => RadonCounty,
    "sblri-blr" => SblriBLR,
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
    return MODELS[name].wren_model
end

"""
    available_posteriordb_models() -> Vector{String}

Return a vector of PosteriorDB model names which WrenCore has implementations for.
"""
WrenCore.available_posteriordb_models() = MODEL_NAMES

end # module
