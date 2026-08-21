module WrenTests

using Test

@testset verbose=true "Wren.jl" begin
    include("expr.jl")
    include("syntax.jl")
    include("submodel.jl")
    include("vecmodel.jl")
    include("draw.jl")
    include("ad.jl")
    include("reactant.jl")
end

end # module WrenTests
