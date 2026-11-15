using Test
using Wren

@testset verbose=true "Wren.jl" begin
    include("nuts.jl")
    include("sampling.jl")
    include("optimization.jl")
    include("generated_quantities.jl")
    include("posteriordb.jl")
end
