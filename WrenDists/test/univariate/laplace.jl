test_dist(
    "Laplace",
    WrenDists.Laplace(1.0, 2.0),
    Distributions.Laplace(1.0, 2.0);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    # Sample point != μ to avoid abs(0) subgradient disagreement across AD backends
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Laplace(x[1], x[2]), 1.5)
        ADTestCase(f, [1.0, 2.0], (), ())
    end,),
)
