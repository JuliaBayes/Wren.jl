module WrenPosteriorDBTests

using Test
using Wren
using WrenCore
using PosteriorDB
using FlexiChains
using StableRNGs
import Enzyme

pdb = PosteriorDB.database()

@testset "sampling from PosteriorDB models matches reference" begin
    @testset for model_name in (
        "arK-arK",
        "gp_pois_regr-gp_regr",
        "kidiq-kidscore_interaction",
        "sblri-blr",
        "eight_schools-eight_schools_centered",
    )
        reference_chain = FlexiChains.from_posteriordb(pdb, model_name)
        model = WrenCore.posteriordb_model(model_name)
        new_chain = nuts(StableRNG(468), model, 10_000, 4; progress=false)

        display(summarystats(new_chain))
        display(summarystats(reference_chain))

        # TODO: actually test that they're the same. I'm not immediately sure how to choose
        # tolerances though.
    end

end

end
