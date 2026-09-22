function batched_parameter_gradient(::Type{D}, a, b, x) where {D}
    return Enzyme.gradient(Reverse, (a, b) -> logdensityof(D(a, b), x), a, b)
end

@testset "NegativeBinomial" begin
    d = WrenDists.NegativeBinomial(2.5, 0.4)
    ref = Distributions.NegativeBinomial(2.5, 0.4)
    f(x) = logdensityof(WrenDists.NegativeBinomial(x[1], x[2]), 3)
    test_dist(
        "NegativeBinomial",
        d,
        ref;
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(ADTestCase(f, [2.5, 0.4], (), ()),),
    )
    @test ForwardDiff.gradient(f, [2.5, 0.4]) ≈
          [sum(inv(2.5 + j) for j in 0:2) + log(0.4), 2.5 / 0.4 - 3 / 0.6]
    @test logdensityof(d, -1) == -Inf
    @test logdensityof(d, 0.5) == -Inf
    @test logdensityof(WrenDists.NegativeBinomial(1e14, 0.9), 0) == 1e14 * log(0.9)
    pointmass = WrenDists.NegativeBinomial(2.5, 1.0)
    @test rand(Xoshiro(468), pointmass) == 0
    @test logdensityof(pointmass, 0) == 0
    @test logdensityof(pointmass, 1) == -Inf

    samples = rand(Xoshiro(468), d, 4096)
    p = Distributions.cdf(ref, 4)
    @test isapprox(
        sum(samples .<= 4) / length(samples),
        p;
        atol=6 * sqrt(p * (1 - p) / length(samples)),
    )
end

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
        @testset "$B" begin
            a, b = [1.0, 1.5, 2.5], [0.2, 0.5, 0.8]
            f(v) = logdensityof(B(v[1:3], v[4:6]), x)
            test_dist(
                string(B),
                B(a, b),
                Distributions.product_distribution(R.(a, b));
                test_batched=false,
                other_ad_testcases=(ADTestCase(f, vcat(a, b), (), ()),),
            )

            # Preserve the sample shape and precision for matrix-valued batches.
            am, bm = reshape(Float32.(a[1:2]), 1, 2), reshape(Float32.(b[1:2]), 1, 2)
            dm = B(am, bm)
            sample = rand(Xoshiro(468), dm)
            @test size(sample) == size(dm) == (1, 2)
            @test logdensityof(dm, sample) ≈ sum(Distributions.logpdf.(R.(am, bm), sample))
            if RUN_REACTANT
                xm = reshape(Float32.(x[1:2]), 1, 2)
                test_dist_reactant(dm, [xm])
                ar, br, xr = Reactant.to_rarray.((am, bm, xm))
                compiled = Reactant.@compile batched_parameter_gradient(B, ar, br, xr)
                grads = compiled(B, ar, br, xr)
                ref = batched_parameter_gradient(B, am, bm, xm)
                @test all(
                    isapprox(Array(g), r; rtol=1e-5, atol=1e-5) for
                    (g, r) in zip(grads, ref)
                )
            end
        end
    end

    @testset "shared shape gradients" begin
        x = [0.2, 0.5, 0.8]
        for f in (
            a -> logdensityof(WrenDists.BatchedBeta(Fill(a, 3), [0.2, 0.5, 0.8]), x),
            b -> logdensityof(WrenDists.BatchedBeta([0.2, 0.5, 0.8], Fill(b, 3)), x),
        )
            @test only(Enzyme.gradient(Reverse, f, 1.0)) ≈ ForwardDiff.derivative(f, 1.0)
        end
    end

    @testset "support endpoints" begin
        for (B, R) in (
                (WrenDists.BatchedGamma, Distributions.Gamma),
                (WrenDists.BatchedWeibull, Distributions.Weibull),
                (WrenDists.BatchedBeta, Distributions.Beta),
            ),
            a in (0.5, 1.0, 2.0)

            @test logdensityof(B([a], [1.0]), [0.0]) ≈ Distributions.logpdf(R(a, 1.0), 0.0)
        end
        d = WrenDists.BatchedNegativeBinomial([2.5, 3.0], [1.0, 0.5])
        @test logdensityof(d, [0.0, 2.0]) ≈
              Distributions.logpdf(Distributions.NegativeBinomial(3.0, 0.5), 2)
        @test logdensityof(d, [1.0, 2.0]) == -Inf
        @test logdensityof(WrenDists.BatchedGamma([2.0], [1.0]), [-1.0]) == -Inf
        f0(x) = logdensityof(WrenDists.BatchedBeta([1.0], [2.0]), x)
        f1(x) = logdensityof(WrenDists.BatchedBeta([2.0], [1.0]), x)
        @test f1([1.0]) ≈ log(2.0)
        @test first(Enzyme.gradient(Reverse, f0, [0.0])) ≈ [-1.0]
        @test first(Enzyme.gradient(Reverse, f1, [1.0])) ≈ [1.0]
        d = WrenDists.BatchedBeta([2.0], [2.0])
        @test logdensityof(d, [NaN]) == logdensityof(WrenDists.Beta(2.0, 2.0), NaN) == -Inf
        if RUN_REACTANT
            test_dist_reactant(d, [[NaN]])
        end
    end
end
