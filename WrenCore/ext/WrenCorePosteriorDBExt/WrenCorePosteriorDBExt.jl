module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/ark.jl")
include("models/bones.jl")
include("models/capture_common.jl")
include("models/capture_m0.jl")
include("models/capture_mb.jl")
include("models/capture_mt.jl")
include("models/capture_mth.jl")
include("models/capture_mtbh.jl")
include("models/dogs.jl")
include("models/dogs_hierarchical.jl")
include("models/dogs_log.jl")
include("models/eight_schools_centered.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/election88.jl")
include("models/hmm_example.jl")
include("models/hmm_gaussian.jl")
include("models/bball_drive_0.jl")
include("models/bball_drive_1.jl")
include("models/iohmm_reg.jl")
include("models/butterfly_occupancy.jl")
include("models/dogs_nonhierarchical.jl")
include("models/sat_2pl.jl")
include("models/irt_common.jl")
include("models/fims_2pl.jl")
include("models/science_grsm.jl")
include("models/timss_gpcm.jl")
include("models/lda_three_docs.jl")
include("models/lda_three_men.jl")
include("models/glmm_poisson.jl")
include("models/glmm1.jl")
include("models/gp_regr.jl")
include("models/irt_2pl.jl")
include("models/kidscore_interaction.jl")
include("models/lsat.jl")
include("models/mh.jl")
include("models/nes_logit.jl")
include("models/rate1.jl")
include("models/sblri_blr.jl")
include("models/survey.jl")

# Maps PosteriorDB posterior names to containers with a `wren_model` property.
const MODELS = Dict(
    "arK-arK" => ArK,
    "bones_data-bones_model" => Bones,
    "M0_data-M0_model" => CaptureM0,
    "Mb_data-Mb_model" => CaptureMb,
    "Mt_data-Mt_model" => CaptureMt,
    "Mth_data-Mth_model" => CaptureMth,
    "Mtbh_data-Mtbh_model" => CaptureMtbh,
    "dogs-dogs" => Dogs,
    "dogs-dogs_hierarchical" => DogsHierarchical,
    "dogs-dogs_log" => DogsLog,
    "eight_schools-eight_schools_centered" => EightSchoolsCentered,
    "GLMM_data-GLMM1_model" => GLMM1,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson,
    "election88-election88_full" => Election88,
    "hmm_example-hmm_example" => HMMExample,
    "hmm_gaussian_simulated-hmm_gaussian" => HMMGaussian,
    "bball_drive_event_0-hmm_drive_0" => BballDrive0,
    "bball_drive_event_1-hmm_drive_1" => BballDrive1,
    "iohmm_reg_simulated-iohmm_reg" => IOHMMReg,
    "butterfly-multi_occupancy" => ButterflyOccupancy,
    "dogs-dogs_nonhierarchical" => DogsNonhierarchical,
    "sat-hier_2pl" => SAT2PL,
    "fims_Aus_Jpn_irt-2pl_latent_reg_irt" => FIMS2PL,
    "science_irt-grsm_latent_reg_irt" => ScienceGRSM,
    "timssAusTwn_irt-gpcm_latent_reg_irt" => TIMSSGPCM,
    "three_docs1200-ldaK2" => LDAThreeDocs,
    "three_men1-ldaK2" => LDAThreeMen.case1,
    "three_men2-ldaK2" => LDAThreeMen.case2,
    "three_men3-ldaK2" => LDAThreeMen.case3,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson,
    "gp_pois_regr-gp_regr" => GPRegr,
    "irt_2pl-irt_2pl" => IRT2PL,
    "kidiq-kidscore_interaction" => KidscoreInteraction,
    "lsat_data-lsat_model" => LSAT,
    "Mh_data-Mh_model" => Mh,
    "nes_logit_data-nes_logit_model" => NESLogit,
    "Rate_1_data-Rate_1_model" => Rate1,
    "sblri-blr" => SblriBLR,
    "Survey_data-Survey_model" => Survey,
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
