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
    include("ad.jl")
end

end # module WrenCoreTests
