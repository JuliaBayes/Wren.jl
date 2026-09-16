module WrenCoreTests

using Test

@testset verbose=true "WrenCore.jl" begin
    include("expr.jl")
    include("syntax.jl")
    include("batch.jl")
    include("aliasing.jl")
    include("submodel.jl")
    include("vecmodel.jl")
    include("draw.jl")
    include("iid.jl")
    include("ad.jl")
    include("predict.jl")

    if !Sys.iswindows()
        # problematic on windows CI, idk why
        include("reactant.jl")
    end
end

end # module WrenCoreTests
