# Continuous positive, truncated at zero
for (μ, σ) in ((0.0, 2.0), (-1.5, 0.7), (1.0, 2.0))
    test_dist(
        "HalfCauchy($μ,$σ)",
        WrenDists.HalfCauchy(μ, σ),
        truncated(Distributions.Cauchy(μ, σ); lower=0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(
            let f(x) = logdensityof(WrenDists.HalfCauchy(x[1], x[2]), 1.0)
                ADTestCase(f, [μ, σ], (), ())
            end,
        ),
    )
end
