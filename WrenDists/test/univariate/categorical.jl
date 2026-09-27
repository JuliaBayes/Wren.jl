test_dist(
    "Categorical",
    WrenDists.Categorical([0.2, 0.3, 0.5]),
    Distributions.Categorical([0.2, 0.3, 0.5]);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Categorical(x), 2)
        ADTestCase(f, [0.2, 0.3, 0.5], (), ())
    end,),
)
