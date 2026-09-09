using WrenDists
using WrenDists: logdensityof
using Distributions
using LinearAlgebra
using Random
using Reactant
using Test

const RNG = Random.MersenneTwister(12345)

function test_reactant(d, x)
    x_ra = Reactant.to_rarray(x)
    f = Reactant.@compile logdensityof(d, x_ra)
    return Float64(f(d, x_ra))
end

@testset "WrenDists" begin
    @testset "Normal" begin
        d_wd = WrenDists.Normal(1.0, 2.0)
        d_ref = Distributions.Normal(1.0, 2.0)
        for x in [-3.0, 0.0, 1.0, 5.0]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        @test isfinite(rand(RNG, d_wd))
    end

    @testset "LogNormal" begin
        d_wd = WrenDists.LogNormal(0.5, 0.3)
        d_ref = Distributions.LogNormal(0.5, 0.3)
        for x in [0.1, 1.0, 2.0, 10.0]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        @test logdensityof(d_wd, -1.0) == -Inf
        @test rand(RNG, d_wd) > 0
    end

    @testset "Exponential" begin
        d_wd = WrenDists.Exponential(3.0)
        d_ref = Distributions.Exponential(3.0)
        for x in [0.0, 0.5, 1.0, 10.0]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        @test logdensityof(d_wd, -1.0) == -Inf
        @test rand(RNG, d_wd) >= 0
    end

    @testset "Weibull" begin
        d_wd = WrenDists.Weibull(1.5, 2.0)
        d_ref = Distributions.Weibull(1.5, 2.0)
        for x in [0.01, 0.5, 1.0, 5.0]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        @test rand(RNG, d_wd) >= 0
    end

    @testset "Uniform" begin
        d_wd = WrenDists.Uniform(1.0, 3.0)
        d_ref = Distributions.Uniform(1.0, 3.0)
        for x in [1.0, 2.0, 3.0]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        @test logdensityof(d_wd, 0.5) == -Inf
        @test logdensityof(d_wd, 3.5) == -Inf
        x = rand(RNG, d_wd)
        @test 1.0 <= x <= 3.0
    end

    @testset "Laplace" begin
        d_wd = WrenDists.Laplace(1.0, 2.0)
        d_ref = Distributions.Laplace(1.0, 2.0)
        for x in [-5.0, 0.0, 1.0, 3.0]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        @test isfinite(rand(RNG, d_wd))
    end

    @testset "Cauchy" begin
        d_wd = WrenDists.Cauchy(1.0, 2.0)
        d_ref = Distributions.Cauchy(1.0, 2.0)
        for x in [-10.0, 0.0, 1.0, 100.0]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        @test isfinite(rand(RNG, d_wd))
    end

    @testset "Gamma" begin
        for (α, θ) in [(2.0, 1.0), (0.5, 3.0), (1.0, 1.0)]
            d_wd = WrenDists.Gamma(α, θ)
            d_ref = Distributions.Gamma(α, θ)
            for x in [0.01, 0.5, 1.0, 5.0]
                @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
            end
        end
        @test rand(RNG, WrenDists.Gamma(2.0, 1.0)) >= 0
    end

    @testset "Beta" begin
        for (α, β) in [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0)]
            d_wd = WrenDists.Beta(α, β)
            d_ref = Distributions.Beta(α, β)
            for x in [0.1, 0.3, 0.5, 0.9]
                @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
            end
        end
        x = rand(RNG, WrenDists.Beta(2.0, 3.0))
        @test 0 <= x <= 1
    end

    @testset "Bernoulli" begin
        d_wd = WrenDists.Bernoulli(0.3)
        d_ref = Distributions.Bernoulli(0.3)
        @test logdensityof(d_wd, 0.0) ≈ logpdf(d_ref, 0)
        @test logdensityof(d_wd, 1.0) ≈ logpdf(d_ref, 1)
        @test logdensityof(d_wd, 0.5) == -Inf
        x = rand(RNG, d_wd)
        @test x == 0.0 || x == 1.0
    end

    @testset "Categorical" begin
        p = [0.2, 0.3, 0.5]
        d_wd = WrenDists.Categorical(p)
        d_ref = Distributions.Categorical(p)
        for k in 1:3
            @test logdensityof(d_wd, Float64(k)) ≈ logpdf(d_ref, k)
        end
        x = rand(RNG, d_wd)
        @test x in [1.0, 2.0, 3.0]
    end

    @testset "Binomial" begin
        d_wd = WrenDists.Binomial(20, 0.4)
        d_ref = Distributions.Binomial(20, 0.4)
        for k in [0, 5, 10, 20]
            @test logdensityof(d_wd, Float64(k)) ≈ logpdf(d_ref, k)
        end
        x = rand(RNG, d_wd)
        @test 0 <= x <= 20
    end

    @testset "Poisson" begin
        d_wd = WrenDists.Poisson(7.0)
        d_ref = Distributions.Poisson(7.0)
        for k in [0, 3, 7, 15]
            @test logdensityof(d_wd, Float64(k)) ≈ logpdf(d_ref, k)
        end
        @test rand(RNG, d_wd) >= 0
    end

    @testset "Geometric" begin
        d_wd = WrenDists.Geometric(0.3)
        d_ref = Distributions.Geometric(0.3)
        for k in [0, 1, 5, 10]
            @test logdensityof(d_wd, Float64(k)) ≈ logpdf(d_ref, k)
        end
        @test rand(RNG, d_wd) >= 0
    end

    @testset "MvNormal" begin
        μ = [1.0, 2.0]
        L = [1.0 0.0; 0.5 1.0]
        Σ = L * L'
        d_wd = WrenDists.MvNormal(μ, L)
        d_ref = Distributions.MvNormal(μ, Σ)
        for x in [[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end

        # Diagonal
        σ = [1.0, 2.0]
        d_wd = WrenDists.MvNormal(μ, Diagonal(σ))
        d_ref = Distributions.MvNormal(μ, Diagonal(σ .^ 2))
        for x in [[0.0, 0.0], [1.0, 2.0]]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end

        # Isotropic
        d_wd = WrenDists.MvNormal(μ, 2.0 * I)
        d_ref = Distributions.MvNormal(μ, 4.0 * I)
        for x in [[0.0, 0.0], [1.0, 2.0]]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end

        @test length(rand(RNG, WrenDists.MvNormal(μ, L))) == 2
    end

    @testset "Dirichlet" begin
        α = [2.0, 3.0, 1.0]
        d_wd = WrenDists.Dirichlet(α)
        d_ref = Distributions.Dirichlet(α)
        for x in [[0.3, 0.5, 0.2], [0.1, 0.1, 0.8], [0.5, 0.25, 0.25]]
            @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
        end
        x = rand(RNG, d_wd)
        @test sum(x) ≈ 1.0
        @test all(x .>= 0)
    end

    @testset "Multinomial" begin
        d_wd = WrenDists.Multinomial(10, [0.2, 0.3, 0.5])
        d_ref = Distributions.Multinomial(10, [0.2, 0.3, 0.5])
        for x in [[2, 3, 5], [0, 0, 10], [4, 4, 2]]
            @test logdensityof(d_wd, Float64.(x)) ≈ logpdf(d_ref, x)
        end
        x = rand(RNG, d_wd)
        @test sum(x) == 10
    end

    include("plaice.jl")

    @testset "Reactant logdensityof" begin
        @testset "Normal" begin
            d = WrenDists.Normal(1.0, 2.0)
            @test test_reactant(d, 0.5) ≈ logdensityof(d, 0.5)
        end

        @testset "LogNormal" begin
            d = WrenDists.LogNormal(0.5, 0.3)
            @test test_reactant(d, 1.5) ≈ logdensityof(d, 1.5)
        end

        @testset "Exponential" begin
            d = WrenDists.Exponential(3.0)
            @test test_reactant(d, 2.0) ≈ logdensityof(d, 2.0)
        end

        @testset "Weibull" begin
            d = WrenDists.Weibull(1.5, 2.0)
            @test test_reactant(d, 1.0) ≈ logdensityof(d, 1.0)
        end

        @testset "Uniform" begin
            d = WrenDists.Uniform(1.0, 3.0)
            @test test_reactant(d, 2.0) ≈ logdensityof(d, 2.0)
        end

        @testset "Laplace" begin
            d = WrenDists.Laplace(1.0, 2.0)
            @test test_reactant(d, 0.5) ≈ logdensityof(d, 0.5)
        end

        @testset "Cauchy" begin
            d = WrenDists.Cauchy(1.0, 2.0)
            @test test_reactant(d, 0.5) ≈ logdensityof(d, 0.5)
        end

        @testset "Gamma" begin
            d = WrenDists.Gamma(2.0, 1.0)
            @test test_reactant(d, 1.5) ≈ logdensityof(d, 1.5)
        end

        @testset "Beta" begin
            d = WrenDists.Beta(2.0, 3.0)
            @test test_reactant(d, 0.4) ≈ logdensityof(d, 0.4)
        end

        @testset "Bernoulli" begin
            d = WrenDists.Bernoulli(0.3)
            @test test_reactant(d, 1.0) ≈ logdensityof(d, 1.0)
        end

        @testset "Categorical" begin
            d = WrenDists.Categorical([0.2, 0.3, 0.5])
            @test test_reactant(d, 2.0) ≈ logdensityof(d, 2.0)
        end

        @testset "Binomial" begin
            d = WrenDists.Binomial(20, 0.4)
            @test test_reactant(d, 8.0) ≈ logdensityof(d, 8.0)
        end

        @testset "Poisson" begin
            d = WrenDists.Poisson(7.0)
            @test test_reactant(d, 5.0) ≈ logdensityof(d, 5.0)
        end

        @testset "Geometric" begin
            d = WrenDists.Geometric(0.3)
            @test test_reactant(d, 3.0) ≈ logdensityof(d, 3.0)
        end

        @testset "MvNormal (diagonal)" begin
            d = WrenDists.MvNormal([1.0, 2.0], Diagonal([1.0, 2.0]))
            @test test_reactant(d, [0.5, 1.5]) ≈ logdensityof(d, [0.5, 1.5])
        end

        @testset "MvNormal (isotropic)" begin
            d = WrenDists.MvNormal([1.0, 2.0], 2.0 * I)
            @test test_reactant(d, [0.5, 1.5]) ≈ logdensityof(d, [0.5, 1.5])
        end

        @testset "Dirichlet" begin
            d = WrenDists.Dirichlet([2.0, 3.0, 1.0])
            @test test_reactant(d, [0.3, 0.5, 0.2]) ≈ logdensityof(d, [0.3, 0.5, 0.2])
        end

        @testset "Multinomial" begin
            d = WrenDists.Multinomial(10, [0.2, 0.3, 0.5])
            @test test_reactant(d, [2.0, 3.0, 5.0]) ≈ logdensityof(d, [2.0, 3.0, 5.0])
        end
    end
end
