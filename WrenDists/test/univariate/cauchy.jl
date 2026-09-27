test_dist(
    "Cauchy",
    WrenDists.Cauchy(1.0, 2.0),
    Distributions.Cauchy(1.0, 2.0);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Cauchy(x[1], x[2]), 1.0)
        ADTestCase(f, [1.0, 2.0], (), ())
    end,),
)
