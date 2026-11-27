module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/ark.jl")
include("models/logaddexp.jl")
include("models/bball_drive_0.jl")
include("models/bball_drive_1.jl")
include("models/bones.jl")
include("models/butterfly_occupancy.jl")
include("models/capture_common.jl")
include("models/capture_m0.jl")
include("models/capture_mb.jl")
include("models/capture_mt.jl")
include("models/capture_mtbh.jl")
include("models/capture_mth.jl")
include("models/dogs.jl")
include("models/dogs_hierarchical.jl")
include("models/dogs_log.jl")
include("models/dogs_nonhierarchical.jl")
include("models/earnings.jl")
include("models/eight_schools.jl")
include("models/election88.jl")
include("models/irt_common.jl")
include("models/fims_2pl.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm1.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/hmm_example.jl")
include("models/hmm_gaussian.jl")
include("models/iohmm_reg.jl")
include("models/irt_2pl.jl")
include("models/kidscore.jl")
include("models/kilpisjarvi.jl")
include("models/lda_pride_prejudice.jl")
include("models/lda_three_docs.jl")
include("models/lda_three_men.jl")
include("models/lsat.jl")
include("models/mesquite.jl")
include("models/mh.jl")
include("models/mnist_rbm10.jl")
include("models/nes.jl")
include("models/nes_logit.jl")
include("models/pilots.jl")
include("models/radon_pooled.jl")
include("models/rates.jl")
include("models/rats_model.jl")
include("models/sat_2pl.jl")
include("models/sblri_blr.jl")
include("models/science_grsm.jl")
include("models/survey.jl")
include("models/timss_gpcm.jl")
include("models/traffic_bym2.jl")
include("models/wells.jl")

# Maps PosteriorDB posterior names to the Wren models.
const MODELS = Dict(
    "arK-arK" => ArK.model,
    "bball_drive_event_0-hmm_drive_0" => BballDrive0.model,
    "bball_drive_event_1-hmm_drive_1" => BballDrive1.model,
    "bones_data-bones_model" => Bones.model,
    "butterfly-multi_occupancy" => ButterflyOccupancy.model,
    "dogs-dogs" => Dogs.model,
    "dogs-dogs_hierarchical" => DogsHierarchical.model,
    "dogs-dogs_log" => DogsLog.model,
    "dogs-dogs_nonhierarchical" => DogsNonhierarchical.model,
    "earnings-earn_height" => Earnings.earn_height,
    "earnings-log10earn_height" => Earnings.log10earn_height,
    "earnings-logearn_height" => Earnings.logearn_height,
    "earnings-logearn_height_male" => Earnings.logearn_height_male,
    "earnings-logearn_interaction" => Earnings.logearn_interaction,
    "earnings-logearn_interaction_z" => Earnings.logearn_interaction_z,
    "earnings-logearn_logheight_male" => Earnings.logearn_logheight_male,
    "eight_schools-eight_schools_centered" => EightSchools.centered,
    "eight_schools-eight_schools_noncentered" => EightSchools.noncentered,
    "election88-election88_full" => Election88.model,
    "fims_Aus_Jpn_irt-2pl_latent_reg_irt" => FIMS2PL.model,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial.model,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson.model,
    "GLMM_data-GLMM1_model" => GLMM1.model,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson.model,
    "gp_pois_regr-gp_regr" => GPRegr.model,
    "hmm_example-hmm_example" => HMMExample.model,
    "hmm_gaussian_simulated-hmm_gaussian" => HMMGaussian.model,
    "iohmm_reg_simulated-iohmm_reg" => IOHMMReg.model,
    "irt_2pl-irt_2pl" => IRT2PL.model,
    "kidiq-kidscore_interaction" => Kidscore.interaction,
    "kidiq-kidscore_momhs" => Kidscore.momhs,
    "kidiq-kidscore_momhsiq" => Kidscore.momhsiq,
    "kidiq-kidscore_momiq" => Kidscore.momiq,
    "kidiq_with_mom_work-kidscore_interaction_c" => Kidscore.interaction_c,
    "kidiq_with_mom_work-kidscore_interaction_c2" => Kidscore.interaction_c2,
    "kidiq_with_mom_work-kidscore_interaction_z" => Kidscore.interaction_z,
    "kidiq_with_mom_work-kidscore_mom_work" => Kidscore.mom_work,
    "kilpisjarvi_mod-kilpisjarvi" => Kilpisjarvi.model,
    "lsat_data-lsat_model" => LSAT.model,
    "M0_data-M0_model" => CaptureM0.model,
    "Mb_data-Mb_model" => CaptureMb.model,
    "mesquite-logmesquite" => Mesquite.logmesquite,
    "mesquite-logmesquite_logva" => Mesquite.logmesquite_logva,
    "mesquite-logmesquite_logvas" => Mesquite.logmesquite_logvas,
    "mesquite-logmesquite_logvash" => Mesquite.logmesquite_logvash,
    "mesquite-logmesquite_logvolume" => Mesquite.logmesquite_logvolume,
    "mesquite-mesquite" => Mesquite.mesquite,
    "Mh_data-Mh_model" => Mh.model,
    "mnist_100-nn_rbm1bJ10" => MNIST100RBM.model,
    "Mt_data-Mt_model" => CaptureMt.model,
    "Mtbh_data-Mtbh_model" => CaptureMtbh.model,
    "Mth_data-Mth_model" => CaptureMth.model,
    "nes1972-nes" => NES.nes1972,
    "nes1976-nes" => NES.nes1976,
    "nes1980-nes" => NES.nes1980,
    "nes1984-nes" => NES.nes1984,
    "nes1988-nes" => NES.nes1988,
    "nes1992-nes" => NES.nes1992,
    "nes1996-nes" => NES.nes1996,
    "nes2000-nes" => NES.nes2000,
    "nes_logit_data-nes_logit_model" => NESLogit.model,
    "pilots-pilots" => Pilots.model,
    "prideprejudice_chapter-ldaK5" => LDAPridePrejudice.chapter,
    "prideprejudice_paragraph-ldaK5" => LDAPridePrejudice.paragraph,
    "radon_all-radon_pooled" => RadonPooled.radon_all,
    "radon_mn-radon_pooled" => RadonPooled.radon_mn,
    "Rate_1_data-Rate_1_model" => Rates.rate1,
    "Rate_2_data-Rate_2_model" => Rates.rate2,
    "Rate_3_data-Rate_3_model" => Rates.rate3,
    "Rate_4_data-Rate_4_model" => Rates.rate4,
    "Rate_5_data-Rate_5_model" => Rates.rate5,
    "rats_data-rats_model" => RatsModel.model,
    "sat-hier_2pl" => SAT2PL.model,
    "sblri-blr" => SblriBLR.model,
    "science_irt-grsm_latent_reg_irt" => ScienceGRSM.model,
    "Survey_data-Survey_model" => Survey.model,
    "three_docs1200-ldaK2" => LDAThreeDocs.model,
    "three_men1-ldaK2" => LDAThreeMen.case1,
    "three_men2-ldaK2" => LDAThreeMen.case2,
    "three_men3-ldaK2" => LDAThreeMen.case3,
    "timssAusTwn_irt-gpcm_latent_reg_irt" => TIMSSGPCM.model,
    "traffic_accident_nyc-bym2_offset_only" => TrafficBYM2.model,
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
