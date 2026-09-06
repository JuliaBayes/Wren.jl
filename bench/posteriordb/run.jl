include("pmproducts.jl")

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

function bench(mod)
    wren_model = mod.wren_model
    turing_model = mod.turing_model

    # Replace this with basic benchmarking.
    # Use DynamicPPL.TestUtils.AD.run_ad (see DynamicPPL code for this)
    # Use Wren.@bmodel for Wren benchmarking
    #
    # We want to basically only benchmark the vectorised forms i.e. `lp` and `lp_and_grad`
    # for Wren, and for DynamicPPL, whatever `run_ad` returns (that only handles vectorised
    # forms).
    #
    # We want to compare the following:
    #
    #  - Primals (both Wren and Turing)
    #  - Gradients (Wren+Enzyme, Wren+Mooncake, Turing+Enzyme, Turing+Mooncake) ONLY reverse mode
    #  - Primals Wren+Reactant
    #  - Gradients Wren+Reactant/Enzyme ONLY reverse mode
    #
    # If any of these throw an error we would like to catch it and represent it in the
    # return value instead of crashing the whole programme.
    display(rand(wren_model))
    display(rand(turing_model))

    # Then return a struct with the results for each model
end

for mod in (ArK,)
    bench(mod)

    # Then pretty print the structs
end
