module WrenTests

using Test

@testset "Wren.jl" begin
    include("expr.jl")
    include("syntax.jl")
    include("submodel.jl")
    include("vecmodel.jl")
    include("draw.jl")
    include("ad.jl")
end

end # module WrenTests
