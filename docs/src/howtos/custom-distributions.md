# [Custom distributions](@id custom-distributions)

Even though WrenDists already provides many commonly used distributions (read more [here](@ref distributions)), we may sometimes still find ourselves working with distributions that we need to define from scratch.

In this example, we'll demonstrate how to write a zero-inflated Poisson distribution that allows us to not only run inference but also to benefit from other important workflow features Wren provides, such as sampling from the prior.

!!! details "Details on zero-inflation"

    To orient ourselves to the task at hand, we will first define the zero-inflated Poisson distribution.
    This distribution models count data with more zeros than a Poisson distribution would predict.
    It assumes that each observation comes from one of two processes: with probability ``π``, the observation is always zero; otherwise, it is drawn from a Poisson distribution with mean ``λ``.
    Because the Poisson process can also produce zeros, an observed zero could come from either process.

    More formally, we can write this distribution as:

    ```math
    P(Y = k) =
    \begin{cases}
    \pi + (1-\pi)e^{-\lambda}, & k = 0, \\
    (1-\pi)\frac{\lambda^k e^{-\lambda}}{k!}, & k \geq 1.
    \end{cases}
    ```

## How to define a new distribution

We need to define a `PoissonZeroInflated` struct to store the parameters and methods for `DensityInterface.logdensityof` (which Wren reexports) and `Base.rand`:

```@example custom-distributions
using Wren
using Wren.Dists: Poisson
using Random: AbstractRNG, Xoshiro
using LogExpFunctions: logaddexp

struct PoissonZeroInflated{T}
    lambda::T
    pi::T

    function PoissonZeroInflated(lambda::T, pi::T) where {T}
        @assert lambda > zero(T) && zero(T) <= pi <= one(T)
        return new{T}(lambda, pi)
    end
end

function Wren.logdensityof(d::PoissonZeroInflated, x::Int)
    log_1m_pi = log1p(-d.pi)
    return if x == 0
        log_pi = log(d.pi)
        log_1m_pi_pois0 = log_1m_pi - d.lambda
        logaddexp(log_pi, log_1m_pi_pois0)
    else
        log_1m_pi + logdensityof(Poisson(d.lambda), x)
    end
end

function Base.rand(rng::AbstractRNG, d::PoissonZeroInflated)
    return rand(rng) < d.pi ? 0 : Int(rand(rng, Poisson(d.lambda)))
end
```

If you are used to working with custom distributions in Turing.jl, you may notice that we are not subtyping Distributions.jl's types or implementing its full interface — all Wren needs to know is how to calculate the log-density and how to draw a sample.

## Usage inside a model

First we simulate zero-inflated counts that depend on some binary covariate (think for example the location where counts were recorded) that we can fit our model to:

```@example custom-distributions
lambda = 4.5
pi = 0.6
dist = PoissonZeroInflated(lambda, pi)
n = 1000
rng = Xoshiro(123)
y = [rand(rng, dist) for _ in 1:n]
nothing # hide
```

Next up is fitting the model.

```@example custom-distributions
using Wren.Dists: Gamma, Uniform
using Enzyme

@wren function poisson_zeroinflated(y)
    lambda ~ Gamma(2, 1)
    pi ~ Uniform(0, 1)
    for i in eachindex(y)
        y[i] ~ PoissonZeroInflated(lambda, pi)
    end
end

prior_model = poisson_zeroinflated(y)
model = prior_model | (; y=y)
chain = nuts(model, 1000, 4; progress=false)
summarystats(chain)
```

We successfully sampled from our custom zero-inflated Poisson distribution and were able to recover the true parameters reasonably well.

## Manual loglikelihoods

!!! danger "`@addloglikelihood` is discouraged"

    We strongly discourage the use of `@addloglikelihood`.
    It should only be considered if the model would otherwise not work (if you run into such a case, please do let us know in [an issue](https://github.com/JuliaBayes/Wren.jl/issues)).

Instead of using the recommended `~` syntax, it is possible to use the [`@addloglikelihood`](@ref addloglike) macro to increment the loglikelihood manually.

Using `@addloglikelihood` means that you will not be able to use Wren's features relating to [Bayesian workflow](@ref workflow).

For example, you will only be able to sample from the prior via `parameters` if you use the `~` syntax:

```@example custom-distributions
parameters(prior_model)
```

Generating posterior predictions via `simulate` and visualising them with [FlexiChains.jl](https://juliabayes.org/FlexiChains.jl/stable/) also requires the `~` syntax:

```@example custom-distributions
using FlexiChains
using CairoMakie

pp_chain = simulate(prior_model, chain)
FlexiChains.Makie.pushforward_hist(pp_chain, @vn(y); observed=y)
```
