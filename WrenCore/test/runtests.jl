module WrenCoreTests

using Test

@testset verbose=true "WrenCore.jl" begin
    include("expr.jl")
    include("dataflow.jl")
    include("syntax.jl")
    include("batch/general.jl")
    include("batch/normal.jl")
    include("batch/bernoulli.jl")
    include("batch/binomial.jl")
    include("batch/poisson.jl")
    include("batch/exponential.jl")
    include("batch/lognormal.jl")
    include("batch/categorical.jl")
    include("batch/tdist.jl")
    include("batch/two_parameter.jl")
    include("batch/typestability.jl")
    include("aliasing.jl")
    include("submodel.jl")
    include("vecmodel.jl")
    include("draw.jl")
    include("ad.jl")
    include("predict.jl")
    include("issues/40.jl")
    include("issues/69.jl")
    include("issues/132.jl")
    include("issues/149.jl")
end

end # module WrenCoreTests
