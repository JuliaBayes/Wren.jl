module WrenCoreTests

using Test

@testset verbose=true "WrenCore.jl" begin
    include("expr.jl")
    include("syntax.jl")
    include("batch/general.jl")
    include("batch/normal.jl")
    include("batch/bernoulli.jl")
    include("batch/binomial.jl")
    include("batch/poisson.jl")
    include("batch/exponential.jl")
    include("batch/lognormal.jl")
    include("batch/categorical.jl")
    # more distributions to come...
    include("aliasing.jl")
    include("submodel.jl")
    include("vecmodel.jl")
    include("draw.jl")
    include("ad.jl")
end

end # module WrenCoreTests
