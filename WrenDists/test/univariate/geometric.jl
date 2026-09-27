test_dist(
    "Geometric",
    WrenDists.Geometric(0.3),
    Distributions.Geometric(0.3);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Geometric(x[1]), 3)
        ADTestCase(f, [0.3], (), ())
    end,),
)
