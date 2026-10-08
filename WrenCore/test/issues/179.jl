module Issue179

using ADTypes: AutoReactant
using Enzyme
using Reactant
using Test
using WrenCore
using WrenDists

# https://github.com/JuliaBayes/Wren.jl/pull/179
# The above fixed an issue where a kwarg was added to Enzyme
# but not the Reactant ext.
@testset "#179 Reactant samples" begin
    @wren function test123(a)
        x ~ Normal(a, 1.0)
    end
    adtype = AutoReactant(AutoEnzyme(; mode=Enzyme.Reverse))
    nuts(test123(1.0), 100, 4; adtype)
end

end
