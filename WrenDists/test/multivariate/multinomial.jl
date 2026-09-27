test_dist(
    "Multinomial",
    WrenDists.Multinomial(10, [0.2, 0.3, 0.5]),
    Distributions.Multinomial(10, [0.2, 0.3, 0.5]);
    plaice_zero_allocs=MULTIVARIATE_PLAICE,
    # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
    skipped_adtypes=["ReverseDiff", "ReverseDiffCpl"],
    other_ad_testcases=(
        let f(x) = logdensityof(WrenDists.Multinomial(10, x), [2, 3, 5])
            ADTestCase(f, [0.2, 0.3, 0.5], (), ("ReverseDiff", "ReverseDiffCpl"))
        end,
    ),
)
