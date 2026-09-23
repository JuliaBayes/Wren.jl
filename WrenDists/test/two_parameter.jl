test_dist(
    "NegativeBinomial",
    WrenDists.NegativeBinomial(2.5, 0.4),
    Distributions.NegativeBinomial(2.5, 0.4);
    plaice_zero_allocs=UNIVARIATE_PLAICE,
    other_ad_testcases=(
        let f(x) = logdensityof(WrenDists.NegativeBinomial(x[1], x[2]), 3)
            ADTestCase(f, [2.5, 0.4], (), ())
        end,
    ),
)

@testset "Batched two-parameter distributions" begin
    for (B, R, x) in (
        (WrenDists.BatchedGamma, Distributions.Gamma, [0.5, 1.5, 2.5]),
        (WrenDists.BatchedBeta, Distributions.Beta, [0.2, 0.5, 0.8]),
        (WrenDists.BatchedWeibull, Distributions.Weibull, [0.5, 1.5, 2.5]),
        (
            WrenDists.BatchedNegativeBinomial,
            Distributions.NegativeBinomial,
            [0.0, 3.0, 8.0],
        ),
    )
        a, b = [1.0, 1.5, 2.5], [0.2, 0.5, 0.8]
        f(v) = logdensityof(B(v[1:3], v[4:6]), x)
        test_dist(
            string(B),
            B(a, b),
            Distributions.product_distribution(R.(a, b));
            test_batched=false,
            other_ad_testcases=(ADTestCase(f, vcat(a, b), (), ()),),
        )
    end
end

@testset "BatchedBeta gradients" begin
    # The transformed samples in test_dist do not reach these finite-density endpoints.
    f(x) = logdensityof(WrenDists.BatchedBeta([1.0, 2.0], [2.0, 1.0]), x)
    @test only(Enzyme.gradient(Reverse, f, [0.0, 1.0])) ≈ [-1.0, 1.0]

    if RUN_REACTANT
        # The zero coefficient at shape one must retain its parameter derivative.
        g(a, b) = logdensityof(WrenDists.BatchedBeta(a, b), [0.25f0])
        parameter_gradient(a, b) = Enzyme.gradient(Reverse, g, a, b)
        a, b = Reactant.to_rarray.(([1.0f0], [1.0f0]))
        compiled = Reactant.@compile parameter_gradient(a, b)
        da, db = compiled(a, b)
        @test Array(da) ≈ [1 + log(0.25f0)]
        @test Array(db) ≈ [1 + log(0.75f0)]
    end
end
