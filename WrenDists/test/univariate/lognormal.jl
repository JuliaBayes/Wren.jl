test_dist(
    "LogNormal",
    WrenDists.LogNormal(0.5, 0.3),
    Distributions.LogNormal(0.5, 0.3);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.LogNormal(x[1], x[2]), 1.5)
        ADTestCase(f, [0.5, 0.3], (), ())
    end,),
)

test_dist(
    "BatchedLogNormal",
    WrenDists.BatchedLogNormal([0.1, 0.5, 1.0], [0.3, 0.7, 1.2]),
    Distributions.product_distribution(
        Distributions.LogNormal.([0.1, 0.5, 1.0], [0.3, 0.7, 1.2]),
    );
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) =
                logdensityof(WrenDists.BatchedLogNormal(x[1:3], x[4:6]), [1.5, 1.5, 1.5])
            ADTestCase(f, [0.1, 0.5, 1.0, 0.3, 0.7, 1.2], (), ())
        end,
    ),
)

@testset "BatchedLogNormal Enzyme rule" begin
    using EnzymeTestUtils
    mu = [0.1, 0.5, 1.0]
    sigma = [0.3, 0.7, 1.2]
    x = [1.5, 2.0, 3.0]
    @testset "reverse" begin
        @testset for Tmu in (Const, Duplicated),
            Tsigma in (Const, Duplicated),
            Tx in (Const, Duplicated)

            test_reverse(
                WrenDists._lognormal_loop,
                Active,
                (mu, Tmu),
                (sigma, Tsigma),
                (x, Tx),
            )
        end
    end
    @testset "forward" begin
        @testset for RT in (Const, DuplicatedNoNeed, Duplicated),
            Tmu in (Const, Duplicated),
            Tsigma in (Const, Duplicated),
            Tx in (Const, Duplicated)

            test_forward(WrenDists._lognormal_loop, RT, (mu, Tmu), (sigma, Tsigma), (x, Tx))
        end
    end
end
