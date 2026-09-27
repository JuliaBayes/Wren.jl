test_dist(
    "Uniform",
    WrenDists.Uniform(1.0, 3.0),
    Distributions.Uniform(1.0, 3.0);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Uniform(x[1], x[2]), 2.0)
        ADTestCase(f, [1.0, 3.0], (), ())
    end,),
)
