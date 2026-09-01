module WrenTests

using Test

@testset verbose=true "Wren.jl" begin
    include("expr.jl")
    include("syntax.jl")
    include("batch.jl")
    include("aliasing.jl")
    include("submodel.jl")
    include("vecmodel.jl")
    include("draw.jl")
    include("ad.jl")

    if !Sys.iswindows()
        # problematic on windows CI, idk why
        include("reactant.jl")
    end
end

end # module WrenTests
