test_dist(
    "Binomial",
    WrenDists.Binomial(20, 0.4),
    Distributions.Binomial(20, 0.4);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Binomial(20, x[1]), 8)
        ADTestCase(f, [0.4], (), ())
    end,),
)

test_dist(
    "BinomialLogit",
    WrenDists.BinomialLogit(20, -0.4),
    Distributions.Binomial(20, inv(1 + exp(0.4)));
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.BinomialLogit(20, x[1]), 8)
        ADTestCase(f, [-0.4], (), ())
    end,),
)

test_dist(
    "BatchedBinomial",
    WrenDists.BatchedBinomial([5, 10, 20], [0.2, 0.5, 0.8]),
    Distributions.product_distribution(
        Distributions.Binomial.([5, 10, 20], [0.2, 0.5, 0.8]),
    );
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) =
                logdensityof(WrenDists.BatchedBinomial([5, 10, 20], x), [1.0, 4.0, 15.0])
            ADTestCase(f, [0.2, 0.5, 0.8], (), ())
        end,
    ),
)

test_dist(
    "BatchedBinomialLogit",
    WrenDists.BatchedBinomialLogit([5, 10, 20], [-1.4, 0.0, 1.4]),
    Distributions.product_distribution(
        Distributions.Binomial.([5, 10, 20], inv.(1 .+ exp.(-[-1.4, 0.0, 1.4]))),
    );
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedBinomialLogit([5, 10, 20], x),
                [1.0, 4.0, 15.0],
            )
            ADTestCase(f, [-1.4, 0.0, 1.4], (), ())
        end,
    ),
)
