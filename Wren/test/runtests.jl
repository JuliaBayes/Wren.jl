module WrenTests

using Test

@testset verbose=true "Wren.jl" begin
    include("sampling.jl")
end

end # module WrenTests
