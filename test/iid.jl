module WrenIIDTests

using Wren
using Test
using LinearAlgebra: Diagonal, I
using Random: Xoshiro
using Plaice: Plaice
using ProbabilityMeasures: ProbabilityMeasures as PM
using ForwardDiff: ForwardDiff
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArray

scalar_logdensity(d, x) = sum(logdensityof(d, xi) for xi in x)
function scalar_link(d, x)
    b = Plaice.to_unconstrained_vec(d)
    results = [Plaice.with_logabsdet_jacobian(b, xi) for xi in x]
    return vec([only(first(r)) for r in results]), sum(last, results)
end

function with_scalar_indexing_disallowed(f)
    saved = get(task_local_storage(), :ScalarIndexing, nothing)
    task_local_storage(:ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    try
        return f()
    finally
        if saved === nothing
            delete!(task_local_storage(), :ScalarIndexing)
        else
            task_local_storage(:ScalarIndexing, saved)
        end
    end
end

@testset "iid.jl" begin
    @testset "univariate $(d)" for d in (
        PM.Normal(0.5, 2.0),
        PM.Normal(0, 1),
        PM.Exponential(1.5),
        PM.LogNormal(0.2, 0.7),
        PM.Uniform(-1.0, 2.0),
        PM.Cauchy(0.0, 1.0),
        PM.Poisson(3.0),
        PM.Bernoulli(0.3),
    )
        m = iid(d, (3, 4))
        @test m != iid(d, (4, 3))
        @test size(m) == (3, 4)
        @test length(m) == 12
        @test sprint(show, m) == "iid($(d), (3, 4))"

        x = rand(Xoshiro(468), m)
        @test size(x) == (3, 4)
        rng = Xoshiro(468)
        expected = [rand(rng, d) for _ in 1:3, _ in 1:4]
        # Implementation detail: `iid` currently draws elementwise in column-major order,
        # so the result matches scalar draws from the same seed. The real requirement is
        # only that each element is drawn from `d`; update this if the implementation
        # changes.
        @test x == expected
        @test eltype(x) == eltype(expected)
        @test all(xi -> PM.insupport(d, xi), x)
        @test logdensityof(m, x) ≈ scalar_logdensity(d, x)

        @testset "Plaice transforms" begin
            to_u = Plaice.to_unconstrained_vec(m)
            from_u = Plaice.from_unconstrained_vec(m)
            @test Plaice.inverse(to_u) == from_u
            @test Plaice.inverse(from_u) == to_u

            expected_y, expected_lj = scalar_link(d, x)
            y, lj = Plaice.with_logabsdet_jacobian(to_u, x)
            @test y isa AbstractVector
            @test y ≈ expected_y
            @test lj ≈ expected_lj
            @test to_u(x) ≈ expected_y

            x_back, lj_back = Plaice.with_logabsdet_jacobian(from_u, y)
            @test size(x_back) == size(x)
            @test x_back ≈ x
            @test lj_back ≈ -expected_lj
            @test from_u(y) ≈ x

            @test Plaice.to_vec(m)(x) == vec(x)
            @test Plaice.from_vec(m)(vec(x)) == x
            @test Plaice.vec_length(m) == 12
            @test Plaice.unconstrained_vec_length(m) == 12
        end
    end

    @testset "density types" begin
        for d in (PM.Normal(0, 1), PM.Normal(0.0f0, 1.0f0), PM.Exponential(2.0f0))
            m = iid(d, (5,))
            x32 = Float32.(rand(m))
            @test logdensityof(m, x32) isa Float32
            @test logdensityof(m, Float64.(x32)) isa Float64
        end
        d = PM.Normal(0.3, 1.7)
        x = Float32.(rand(iid(d, (7,))))
        @test logdensityof(iid(d, (7,)), x) ≈ scalar_logdensity(d, x)
        @test isnan(logdensityof(iid(PM.Normal(0.0, -1.0), (3,)), [0.1, 0.2, 0.3]))
    end

    @testset "multivariate draws" begin
        d = PM.MvNormal([0.0, 1.0], Diagonal([1.0, 2.0]))
        m = iid(d, (5,))
        @test size(m) == (2, 5)
        x = rand(m)
        @test size(x) == (2, 5)
        @test logdensityof(m, x) ≈ sum(logdensityof(d, c) for c in eachcol(x))

        m2 = iid(d, (3, 4))
        x2 = rand(Xoshiro(4), m2)
        @test size(x2) == (2, 3, 4)
        rng = Xoshiro(4)
        expected = reshape(hcat([rand(rng, d) for _ in 1:12]...), 2, 3, 4)
        # Implementation detail (see the univariate testset above).
        @test x2 == expected
        @test eltype(x2) == Float64
        @test logdensityof(m2, x2) ≈
              sum(logdensityof(d, x2[:, i, j]) for i in 1:3, j in 1:4)

        L = [1.0 0.0; 0.5 2.0]
        for factor in (I, 2.5 * I, Diagonal([1.0, 2.0]), L)
            dk = PM.MvNormal([0.0, 1.0], factor)
            mk = iid(dk, (6,))
            xk = rand(mk)
            @test logdensityof(mk, xk) ≈ sum(logdensityof(dk, c) for c in eachcol(xk))
        end

        @test Plaice.from_unconstrained_vec(m) == Plaice.Reshape((2, 5))
        @test Plaice.to_unconstrained_vec(m) == Plaice.Vec((2, 5))
        @test Plaice.from_vec(m)(vec(x)) == x
        y, lj = Plaice.with_logabsdet_jacobian(Plaice.to_unconstrained_vec(m), x)
        @test y == vec(x)
        @test iszero(lj)
    end

    @testset "Multinomial draws" begin
        d = PM.Multinomial(5, [0.2, 0.8])
        m = iid(d, (2, 3))
        x = rand(m)
        @test size(x) == (2, 2, 3)
        @test all(sum(x; dims=1) .== 5)
        @test logdensityof(m, x) ≈ sum(logdensityof(d, x[:, i, j]) for i in 1:2, j in 1:3)
        empty_model = iid(d, (0,))
        @test iszero(logdensityof(empty_model, rand(empty_model)))
    end

    @testset "Normal scale" begin
        for (d, x) in (
            (PM.Normal(0, 1.5), [1, 2]),
            (PM.Normal(0, 1.5f0), [1, 2]),
            (PM.Normal(0.0f0, 1.5), Float32[1, 2]),
        )
            actual = logdensityof(iid(d, (2,)), x)
            expected = scalar_logdensity(d, x)
            @test actual ≈ expected
            @test typeof(actual) == typeof(expected)
        end
        x = [1.0, 2.0]
        actual =
            ForwardDiff.derivative(σ -> logdensityof(iid(PM.Normal(0.0, σ), (2,)), x), 2.0)
        # d/dσ log p(x | σ) = -N/σ + sum(x²)/σ³.
        @test actual ≈ -0.375

        @wren function scale_model(y)
            σ ~ PM.Exponential(1.0)
            @observe y ~ iid(PM.Normal(0.0, σ), (length(y),))
        end
        vm = VecModel(scale_model(x), AutoForwardDiff())
        value, grad = lp_and_grad(vm, [log(2.0)])
        expected_lp(t) =
            logdensityof(PM.Exponential(1.0), exp(t)) +
            scalar_logdensity(PM.Normal(0.0, exp(t)), x) +
            t
        @test value ≈ expected_lp(log(2.0))
        @test only(grad) ≈ -1.75
    end

    @testset "sample shapes" begin
        for d in (PM.Normal(), PM.Exponential(1.0))
            @test_throws DimensionMismatch logdensityof(iid(d, (3,)), zeros(2))
            @test_throws DimensionMismatch logdensityof(iid(d, (2, 3)), zeros(3, 2))
        end
        d = PM.MvNormal(zeros(2), I)
        @test_throws DimensionMismatch logdensityof(iid(d, (3,)), zeros(3, 2))
        @test_throws DimensionMismatch logdensityof(iid(d, (3,)), zeros(2, 2))
        m = independent([PM.Normal(), PM.Normal()])
        @test_throws DimensionMismatch logdensityof(m, zeros(1))
        @test_throws DimensionMismatch logdensityof(m, zeros(2, 2))
    end

    @testset "empty multivariate samples" begin
        for T in (Float32, Float64), dims in ((0,), (3, 0), (0, 2))
            d = PM.MvNormal(zeros(T, 2), I)
            m = iid(d, dims)
            rng = Xoshiro(42)
            x = rand(rng, m)
            @test size(x) == (2, dims...)
            @test eltype(x) == T
            @test rand(rng) == rand(Xoshiro(42))
            ld = logdensityof(m, x)
            @test iszero(ld)
            @test ld isa T
        end
    end

    @testset "empty univariate samples" begin
        for d in (PM.Normal(0.5f0, 2.0f0), PM.Normal(0, 1), PM.Exponential(1.5)),
            dims in ((0,), (3, 0))

            m = iid(d, dims)
            x = rand(Xoshiro(42), m)
            @test size(x) == dims
            @test eltype(x) == eltype(d)
            ld = logdensityof(m, x)
            @test iszero(ld)
            @test ld isa eltype(d)
            @test iszero(logdensityof(m, Float32.(x)))
        end
        @wren function empty_plate(y, N)
            @plate for i in 1:N
                @observe y[i] ~ PM.Exponential(2.0)
            end
        end
        @test iszero(loglikelihood(rand_with_logp(empty_plate(Float64[], 0))))
    end

    @testset "MvNormal result types" begin
        for T in (Float32, Float64, BigFloat)
            for L in (I, T(2) * I, Diagonal(T[1, 2]), T[1 0; 0.5 2])
                d = PM.MvNormal(T[0, 1], L)
                x = T[0.5 1; 1.5 2]
                actual = logdensityof(iid(d, (2,)), x)
                expected = sum(logdensityof(d, c) for c in eachcol(x))
                @test actual ≈ expected
                @test typeof(actual) == typeof(expected)
            end
        end
    end

    @testset "arrays without scalar indexing" begin
        with_scalar_indexing_disallowed() do
            @test_throws ErrorException JLArray([1.0])[1]
            for d in
                (PM.Normal(0.5f0, 2.0f0), PM.Exponential(1.5f0), PM.Uniform(-1.0f0, 2.0f0))
                m = iid(d, (4, 3))
                x = rand(m)
                xg = JLArray(x)
                @test logdensityof(m, xg) ≈ logdensityof(m, x)

                to_u = Plaice.to_unconstrained_vec(m)
                from_u = Plaice.from_unconstrained_vec(m)
                y, lj = Plaice.with_logabsdet_jacobian(to_u, xg)
                y_cpu, lj_cpu = Plaice.with_logabsdet_jacobian(to_u, x)
                @test y isa JLArray
                @test Array(y) ≈ y_cpu
                @test lj ≈ lj_cpu
                x_back, lj_back = Plaice.with_logabsdet_jacobian(from_u, y)
                @test x_back isa JLArray
                @test Array(x_back) ≈ x
                @test lj_back ≈ -lj_cpu
            end
            mv = iid(PM.MvNormal([0.0f0, 1.0f0], Diagonal([1.0f0, 2.0f0])), (5,))
            xmv = rand(mv)
            @test logdensityof(mv, JLArray(xmv)) ≈ logdensityof(mv, xmv)
        end
    end

    @testset "independent" begin
        rates = [0.5, 1.0, 2.0]
        dists = PM.Exponential.(rates)
        m = independent(dists)
        @test sprint(show, m) == "independent($(dists))"
        @test m == independent(copy(dists))
        @test m != independent(PM.Exponential.([0.5, 1.0, 3.0]))
        repeated = independent(fill(PM.Exponential(2.0), 3))
        @test logdensityof(iid(PM.Exponential(2.0), (3,)), [0.2, 0.7, 1.3]) ≈
              logdensityof(repeated, [0.2, 0.7, 1.3])
        @test length(m) == 3
        x = rand(Xoshiro(6), m)
        rng = Xoshiro(6)
        # Implementation detail (see the univariate testset above).
        @test x == [rand(rng, d) for d in dists]
        @test size(x) == (3,)
        @test logdensityof(m, x) ≈ sum(logdensityof.(dists, x))
        to_u = Plaice.to_unconstrained_vec(m)
        from_u = Plaice.from_unconstrained_vec(m)
        y, lj = Plaice.with_logabsdet_jacobian(to_u, x)
        @test y ≈ log.(x)
        @test lj ≈ -sum(log, x)
        x_back, lj_back = Plaice.with_logabsdet_jacobian(from_u, y)
        @test x_back ≈ x
        @test lj_back ≈ sum(log, x)

        udists = [PM.Uniform(0.0, 1.0), PM.Uniform(-2.0, 3.0)]
        um = independent(udists)
        ux = [0.25, 1.0]
        uy, ulj = Plaice.with_logabsdet_jacobian(Plaice.to_unconstrained_vec(um), ux)
        expected = [
            Plaice.with_logabsdet_jacobian(Plaice.to_unconstrained_vec(di), xi) for
            (di, xi) in zip(udists, ux)
        ]
        @test uy ≈ [only(first(e)) for e in expected]
        @test ulj ≈ sum(last, expected)

        @test_throws ArgumentError independent([PM.MvNormal([0.0], I)])
    end

    @testset "different component distributions" begin
        dists = [PM.Normal(0.5, 2.0), PM.Exponential(2.0), PM.Uniform(-2.0, 3.0)]
        m = independent(dists)
        x = [-0.3, 1.4, 0.7]
        @test logdensityof(m, x) ≈ sum(logdensityof(d, xi) for (d, xi) in zip(dists, x))

        expected = [scalar_link(d, [xi]) for (d, xi) in zip(dists, x)]
        y, lj = Plaice.with_logabsdet_jacobian(Plaice.to_unconstrained_vec(m), x)
        @test y ≈ [only(first(r)) for r in expected]
        @test lj ≈ sum(last, expected)
        x_back, inv_lj = Plaice.with_logabsdet_jacobian(Plaice.from_unconstrained_vec(m), y)
        @test x_back ≈ x
        @test inv_lj ≈ -lj
    end

    @testset "model integration" begin
        @wren function latent_model(d, N)
            x ~ iid(d, (N,))
            return x
        end

        @testset "latent $(d)" for d in (
            PM.Normal(0.5, 2.0),
            PM.Exponential(2.0),
            PM.Uniform(0.0, 1.0),
        )
            model = latent_model(d, 4)
            draw = rand(Xoshiro(468), model)
            @test size(draw[@vn(x)]) == (4,)

            vm = VecModel(model, AutoForwardDiff())
            params = [-0.7, 0.2, 1.1, 1.8]
            @test length(rand(Xoshiro(469), vm)) == 4

            from_u = Plaice.from_unconstrained_vec(d)
            function expected_lp(params)
                return sum(params) do p
                    x, lj = Plaice.with_logabsdet_jacobian(from_u, [p])
                    logdensityof(d, x) + lj
                end
            end
            @test lp(vm, params) ≈ expected_lp(params)
            value, grad = lp_and_grad(vm, params)
            @test value ≈ expected_lp(params)
            @test grad ≈ ForwardDiff.gradient(expected_lp, params)
        end

        @testset "observed" begin
            @wren function obs_model(y, d)
                @observe y ~ iid(d, size(y))
            end
            y = [0.3 1.2; 2.5 0.7]
            d = PM.Exponential(1.5)
            accs = rand_with_logp(obs_model(y, d))
            @test loglikelihood(accs) ≈ scalar_logdensity(d, y)
        end

        @testset "@plate with a shared distribution" begin
            @wren function plate_model(y, N)
                @plate for i in 1:N
                    @observe y[i] ~ PM.Exponential(2.0)
                end
            end
            y = [0.1, 0.4, 1.3]
            accs = rand_with_logp(plate_model(y, 3))
            @test loglikelihood(accs) ≈ scalar_logdensity(PM.Exponential(2.0), y)
            @test Wren.batched(1:3, PM.Exponential(2.0)) == iid(PM.Exponential(2.0), (3,))
        end

        @testset "@plate with indexed parameters" begin
            @wren function hetero_model(y, rates, N)
                @plate for i in 1:N
                    @observe y[i] ~ PM.Exponential(rates[i])
                end
            end
            y = [0.1, 0.4, 1.3]
            rates = [0.5, 1.0, 2.0]
            accs = rand_with_logp(hetero_model(y, rates, 3))
            @test loglikelihood(accs) ≈ sum(logdensityof.(PM.Exponential.(rates), y))
        end
    end
end

end # module WrenIIDTests
