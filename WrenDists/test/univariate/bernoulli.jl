test_dist(
    "Bernoulli",
    WrenDists.Bernoulli(0.3),
    Distributions.Bernoulli(0.3);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Bernoulli(x[1]), 1)
        ADTestCase(f, [0.3], (), ())
    end,),
)

test_dist(
    "BernoulliLogit",
    WrenDists.BernoulliLogit(-0.4),
    Distributions.BernoulliLogit(-0.4);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.BernoulliLogit(x[1]), 1)
        ADTestCase(f, [-0.4], (), ())
    end,),
)

test_dist(
    "BatchedBernoulli",
    WrenDists.BatchedBernoulli([0.2, 0.5, 0.8]),
    Distributions.product_distribution(Distributions.Bernoulli.([0.2, 0.5, 0.8]));
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedBernoulli(x), [1.0, 0.0, 1.0])
            ADTestCase(f, [0.2, 0.5, 0.8], (), ())
        end,
    ),
)

test_dist(
    "BatchedBernoulliLogit",
    WrenDists.BatchedBernoulliLogit([-1.4, 0.0, 1.4]),
    Distributions.product_distribution(Distributions.BernoulliLogit.([-1.4, 0.0, 1.4]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedBernoulliLogit(x), [1.0, 0.0, 1.0])
            ADTestCase(f, [-1.4, 0.0, 1.4], (), ())
        end,
    ),
)
