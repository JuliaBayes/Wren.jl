test_dist(
    "Weibull",
    WrenDists.Weibull(1.5, 2.0),
    Distributions.Weibull(1.5, 2.0);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Weibull(x[1], x[2]), 1.0)
        ADTestCase(f, [1.5, 2.0], (), ())
    end,),
)

test_dist(
    "BatchedWeibull",
    WrenDists.BatchedWeibull([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    Distributions.product_distribution(
        Distributions.Weibull.([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    );
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedWeibull(x[1:3], x[4:6]), [0.5, 1.5, 2.5])
            ADTestCase(f, [1.0, 1.5, 2.5, 0.2, 0.5, 0.8], (), ())
        end,
    ),
)
