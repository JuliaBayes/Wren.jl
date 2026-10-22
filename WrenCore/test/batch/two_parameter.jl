module WrenBatchTwoParameterTests

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists
import Enzyme

@wren function paired_batch(D, y)
    a ~ WrenCore.iid(WrenDists.LogNormal(), 3)
    b ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
    @batch for i in 1:3
        @observe y[i] ~ D(a[i], b[i])
        @observe y[i] ~ D(a[i], b[1])
        @observe y[i] ~ D(a[1], b[i])
    end
end

@wren function paired_nobatch(D, y)
    a ~ WrenCore.iid(WrenDists.LogNormal(), 3)
    b ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
    for i in 1:3
        @observe y[i] ~ D(a[i], b[i])
        @observe y[i] ~ D(a[i], b[1])
        @observe y[i] ~ D(a[1], b[i])
    end
end

@wren function default_batch(D, y)
    a ~ WrenCore.iid(WrenDists.LogNormal(), 3)
    @batch for i in 1:3
        @observe y[i] ~ D(a[i])
    end
end

@wren function default_nobatch(D, y)
    a ~ WrenCore.iid(WrenDists.LogNormal(), 3)
    for i in 1:3
        @observe y[i] ~ D(a[i])
    end
end

@testset "batch: two-parameter distributions" begin
    @testset "$D" for D in (
        WrenDists.Gamma,
        WrenDists.Beta,
        WrenDists.Weibull,
        WrenDists.NegativeBinomial,
    )
        y = D === WrenDists.NegativeBinomial ? [0.0, 3.0, 8.0] : [0.2, 0.5, 0.8]
        test_model_equivalence(
            paired_batch(D, y),
            paired_nobatch(D, y),
            AutoEnzyme(; mode=Reverse),
        )
        if D !== WrenDists.Beta
            test_model_equivalence(
                default_batch(D, y),
                default_nobatch(D, y),
                AutoEnzyme(; mode=Reverse),
            )
        end
    end
end

end # module WrenBatchTwoParameterTests
