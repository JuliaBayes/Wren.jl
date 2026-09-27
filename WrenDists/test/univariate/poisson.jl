test_dist(
    "Poisson",
    WrenDists.Poisson(7.0),
    Distributions.Poisson(7.0);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Poisson(x[1]), 7)
        ADTestCase(f, [7.0], (), ())
    end,),
)

test_dist(
    "PoissonLog",
    WrenDists.PoissonLog(log(7.0)),
    Distributions.Poisson(7.0);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.PoissonLog(x[1]), 7)
        ADTestCase(f, [log(7.0)], (), ())
    end,),
)

test_dist(
    "BatchedPoisson",
    WrenDists.BatchedPoisson([1.0, 3.0, 7.0]),
    Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedPoisson(x), [1.0, 2.0, 8.0])
            ADTestCase(f, [1.0, 3.0, 7.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedPoissonLog",
    WrenDists.BatchedPoissonLog(log.([1.0, 3.0, 7.0])),
    Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedPoissonLog(x), [1.0, 2.0, 8.0])
            ADTestCase(f, log.([1.0, 3.0, 7.0]), (), ())
        end,
    ),
)
