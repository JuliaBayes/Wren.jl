module WrenCoreTests

using Test

@testset verbose=true "WrenCore.jl" begin
    include("expr.jl")
    include("syntax.jl")
    include("batch/general.jl")
    include("batch/normal.jl")
    # more distributions to come...
    include("aliasing.jl")
    include("submodel.jl")
    include("vecmodel.jl")
    include("draw.jl")
    include("ad.jl")
end

end # module WrenCoreTests
