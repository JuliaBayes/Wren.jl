test_dist(
    "Exponential",
    WrenDists.Exponential(3.0),
    Distributions.Exponential(3.0);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Exponential(x[1]), 1.0)
        ADTestCase(f, [3.0], (), ())
    end,),
)

test_dist(
    "BatchedExponential",
    WrenDists.BatchedExponential([0.5, 1.0, 2.0]),
    Distributions.product_distribution(Distributions.Exponential.([0.5, 1.0, 2.0]));
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedExponential(x), [0.5, 0.5, 0.5])
            ADTestCase(f, [0.5, 1.0, 2.0], (), ())
        end,
    ),
)
