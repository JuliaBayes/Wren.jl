module WrenTests

using Test

@testset verbose=true "Wren.jl" begin
    include("predict.jl")
end

end # module WrenTests
