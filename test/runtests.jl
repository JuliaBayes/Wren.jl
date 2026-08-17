module WrenTests

using Test

@testset "Wren.jl" begin
    include("expr.jl")
    include("submodel.jl")
end

end # module WrenTests
