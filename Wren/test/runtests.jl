using Test
using Wren

@testset verbose=true "Wren.jl" begin
    include("nuts.jl")
    include("posteriordb.jl")
end
