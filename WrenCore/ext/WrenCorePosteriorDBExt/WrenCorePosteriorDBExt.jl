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
include("models/radon_county_intercept_all.jl")
include("models/radon_county_intercept_mn.jl")
include("models/radon_hierarchical_intercept_noncentered_all.jl")
include("models/radon_hierarchical_intercept_noncentered_mn.jl")
include("models/radon_partially_pooled_noncentered_all.jl")
include("models/radon_partially_pooled_noncentered_mn.jl")
include("models/radon_variable_intercept_noncentered_all.jl")
include("models/radon_variable_intercept_noncentered_mn.jl")
include("models/radon_variable_intercept_slope_noncentered_all.jl")
include("models/radon_variable_intercept_slope_noncentered_mn.jl")
include("models/radon_variable_slope_noncentered_all.jl")
include("models/radon_variable_slope_noncentered_mn.jl")
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
    "radon_all-radon_county_intercept" => RadonCountyInterceptAll,
    "radon_all-radon_hierarchical_intercept_noncentered" => RadonHierarchicalInterceptNoncenteredAll,
    "radon_all-radon_partially_pooled_noncentered" => RadonPartiallyPooledNoncenteredAll,
    "radon_all-radon_variable_intercept_noncentered" => RadonVariableInterceptNoncenteredAll,
    "radon_all-radon_variable_intercept_slope_noncentered" => RadonVariableInterceptSlopeNoncenteredAll,
    "radon_all-radon_variable_slope_noncentered" => RadonVariableSlopeNoncenteredAll,
    "radon_mn-radon_county_intercept" => RadonCountyInterceptMn,
    "radon_mn-radon_hierarchical_intercept_noncentered" => RadonHierarchicalInterceptNoncenteredMn,
    "radon_mn-radon_partially_pooled_noncentered" => RadonPartiallyPooledNoncenteredMn,
    "radon_mn-radon_variable_intercept_noncentered" => RadonVariableInterceptNoncenteredMn,
    "radon_mn-radon_variable_intercept_slope_noncentered" => RadonVariableInterceptSlopeNoncenteredMn,
    "radon_mn-radon_variable_slope_noncentered" => RadonVariableSlopeNoncenteredMn,
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
