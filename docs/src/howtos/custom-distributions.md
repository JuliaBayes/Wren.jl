# [Custom distributions](@id custom-distributions)

Even though the Julia ecosystem[^1] already provides a rich set of distributions we can directly use with Wren, we may sometimes still have to work with distributions that we need to define ourselves.

In this example, we'll demonstrate how to write a zero-inflated Poisson distribution[^2] that allows us to not only run inference but also to benefit from other important workflow features Wren provides, such as sampling from the prior.

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

If you are used to working with custom distributions in Turing.jl, you may notice that we are not implementing the `Distributions` interface – all Wren needs to know is how to calculate the logdensity and how to draw a sample.

## Usage inside a model

First we simulate zero-inflated counts that depend on some binary covariate (think for example the location where counts were recorded) that we can fit our model to:

```@example custom-distributions
# Distributions per covariate group
lambdas = [4.5, 7.0]
pis = [0.6, 0.2]
dists = [PoissonZeroInflated(lambda, pi) for (lambda, pi) in zip(lambdas, pis)]

n = 1000
x = rand([1, 2], n)
y = rand.(dists[x])
nothing # hide
```

Next up is fitting the model.
The `y` argument declares the shape of the observation array; its values are ignored once we condition on the data (see [Arrays and structs](@ref)).

```@example custom-distributions
@wren function poisson_zeroinflated(x, y)
    k = maximum(x)
    @iid lambda ~ Gamma(2, 1) size=k
    @iid pi ~ Uniform(0, 1) size=k
    for i in eachindex(x)
        y[i] ~ PoissonZeroInflated(lambda[x[i]], pi[x[i]])
    end
end

prior_model = poisson_zeroinflated(x, y)
model = prior_model | VarNamedTuple(y=y)
chain = nuts(model, 1000, 4; progress=false)
summarystats(chain)
```

We successfully sampled from our custom zero-inflated Poisson distribution and were able to recover the true parameters reasonably well.

## Manual loglikelihoods

Instead of using the recommended `~` syntax, you may also use the `@addloglikelihood` macro to increment the ... well, loglikelihood manually.
Be aware that this sacrifices many powerful features of Wren relating to [Bayesian workflow](@ref).

For example, the model using the `~` syntax supports sampling from the prior via `rand`:

```@example custom-distributions
rand(prior_model)
```

... and posterior predictions via `predict` that play nicely with the plotting functions provided by [FlexiChains.jl](https://juliabayes.org/FlexiChains.jl/stable/):

```@example custom-distributions
using CairoMakie

pp_chain = predict(prior_model, chain)
FlexiChains.Makie.pushforward_hist(pp_chain, @vn(y); observed=y)
```

[^1]: See for example [MeasureBase.jl](https://github.com/JuliaMath/MeasureBase.jl). Wren also works with [Distributions.jl](https://github.com/juliastats/distributions.jl) but note that these distributions may not be GPU-compatible.

[^2]: A zero-inflated Poisson distribution accounts for zero counts that would not by described by a standard Poisson by mixing a Bernoulli and Poisson component.
