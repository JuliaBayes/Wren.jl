# [Custom distributions](@id custom-distributions)

Even though WrenDists already provides many commonly used distributions (read more [here](@ref distributions)), we may sometimes still find ourselves working with distributions that we need to define from scratch.

In this example, we'll demonstrate how to write a zero-inflated Poisson distribution that allows us to not only run inference but also to benefit from other important workflow features Wren provides, such as sampling from the prior.

!!! note "Zero-inflation"
    A zero-inflated Poisson distribution accounts for zero counts that would not by described by a standard Poisson by mixing a Dirac and Poisson component.

## How to define a new distribution

We need to define `PoissonZeroInflated` to store the parameters and methods for `Wren.logdensityof` and `Base.rand`:

```@example custom-distributions
using Wren
using Wren.Dists
using FlexiChains
using Enzyme
using Random: AbstractRNG
using LogExpFunctions: logaddexp

struct PoissonZeroInflated{T}
    lambda::T
    pi::T

    function PoissonZeroInflated(lambda::T, pi::T) where T
        @assert lambda > 0 && 0 <= pi <= 1
        return new{T}(lambda, pi)
    end
end

function Wren.logdensityof(d::PoissonZeroInflated, x::Int)
    log_1m_pi = log1p(-d.pi)
    if x == 0
        log_pi = log(d.pi)
        log_1m_pi_pois0 = log_1m_pi - d.lambda
        return logaddexp(log_pi, log_1m_pi_pois0)
    else
        return log_1m_pi + logdensityof(Poisson(d.lambda), x)
    end
end

function Base.rand(rng::AbstractRNG, d::PoissonZeroInflated)
    return rand(rng) < d.pi ? 0 : Int(rand(rng, Poisson(d.lambda)))
end
```

If you are used to working with custom distributions in Turing.jl, you may notice that we are not implementing the `Distributions` interface — all Wren needs to know is how to calculate the logdensity and how to draw a sample.

## Usage inside a model

First we simulate zero-inflated counts that depend on some binary covariate (think for example the location where counts were recorded) that we can fit our model to:

```@example custom-distributions
lambda = 4.5
pi = 0.6
dist = PoissonZeroInflated(lambda, pi)
n = 1000
y = rand(dist, n)
nothing # hide
```

Next up is fitting the model.

```@example custom-distributions
@wren function poisson_zeroinflated(y)
    lambda ~ Gamma(2, 1)
    pi ~ Uniform(0, 1)
    for i in eachindex(y)
        y[i] ~ PoissonZeroInflated(lambda, pi)
    end
end

prior_model = poisson_zeroinflated(y)
model = prior_model | VarNamedTuple(y=y)
chain = nuts(model, 1000, 4; progress=false)
summarystats(chain)
```

We successfully sampled from our custom zero-inflated Poisson distribution and were able to recover the true parameters reasonably well.

## Manual loglikelihoods

!!! danger "We do NOT recommend this"
    We strongly discourage the use of `@addloglikelihood`. It should only be considered if the model would otherwise not work (if you run into such a case, please let us know in an issue).

Instead of using the recommended `~` syntax, it is possible the `@addloglikelihood` macro to increment the ... well, loglikelihood manually.
    
Using `@addloglikelihood` means that you will not be able to use Wren's powerful features relating to [Bayesian workflow](@ref).

For example, you will only be able to sample from the prior via `rand` if you use the `~` syntax:

```@example custom-distributions
rand(prior_model)
```

Generating posterior predictions via `predict` and visualising them with [FlexiChains.jl](https://juliabayes.org/FlexiChains.jl/stable/) also requires the `~` syntax:

```@example custom-distributions
using CairoMakie

pp_chain = predict(prior_model, chain)
FlexiChains.Makie.pushforward_hist(pp_chain, @vn(y); observed=y)
```
