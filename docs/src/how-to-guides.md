# [How-to guides](@id how-to-guides)

## Custom distributions

Even though the Julia ecosystem[^1] already provides a rich set of distributions we can directly use with Wren, we may sometimes still have to work with distributions that we need to define ourselves.

In this example, we'll demonstrate how to write a zero-inflated Poisson distribution[^2] that allows us to not only run inference but also to benefit from other important workflow features Wren provides, such as sampling from the prior.

### Set up

We need to define `PoissonZeroInflated` to store the parameters and methods for `Wren.logdensityof` and `Base.rand`:

```julia
using ProbabilityMeasures
using Wren
using WrenCore: iid
using Enzyme
using Random: AbstractRNG
using LogExpFunctions: logaddexp

struct PoissonZeroInflated
    lambda::Float64
    pi::Float64

    function PoissonZeroInflated(lambda, pi)
        @assert lambda > 0 && 0 <= pi <= 1
        return new(lambda, pi)
    end
end

function Wren.logdensityof(d::PoissonZeroInflated, x::Int)
    if x == 0
        log_pi = log(d.pi)
        log_1m_pi_pois0 = log(1 - d.pi) - d.lambda
        return logaddexp(log_pi, log_1m_pi_pois0)
    else
        return log(1 - d.pi) + logdensityof(Poisson(d.lambda), x)
    end
end

function Base.rand(rng::AbstractRNG, d::PoissonZeroInflated)
    return rand(rng) < d.pi ? 0 : Int(rand(rng, Poisson(d.lambda)))
end
```

If you are used to working with custom distributions in Turing.jl, you may notice that we are not implementing the `Distributions` interface – all Wren needs to know is how to calculate the logdensity and how to draw a sample.

### Naive model

First we simulate zero-inflated counts that depend on some binary covariate (think for example the location where counts were recorded) that we can fit our model to:

```julia
# Distributions per covariate group
lambdas = [4.5, 7.0]
pis = [0.6, 0.2]
dists = [PoissonZeroInflated(lambda, pi) for (lambda, pi) in zip(lambdas, pis)]

n = 100
x = rand([1, 2], n)
y = rand.(dists[x]);
```

Next up is fitting the model.
The `y = zeros(Int, length(x))` line declares the shape of the observation array; its values are ignored once we condition on the data (see [Arrays and structs](@ref)).

```julia
@wren function poisson_zeroinflated_naive(x)
    k = maximum(x)
    lambda ~ iid(Gamma(2, 1), (k,))
    pi ~ iid(Uniform(0, 1), (k,))
    for i in eachindex(x)
        @observe y[i] ~ PoissonZeroInflated(lambda[x[i]], pi[x[i]])
    end
end

prior_model_naive = poisson_zeroinflated_naive(x)
model_naive = prior_model_naive | VarNamedTuple(y=y)
@bmodel model_naive
```

TODO: should we run MCMC and recover the parameters? Is this relevant for this how-to? Depends on how focused we want to keep it.

### Optimised model

TODO: Showcase batched distribution? Could then link to another how-to guide that describes this in more detail – here we just apply it?

## addloglikelihood macro

TODO: Add more detail here, describe limitations, refer to the operations above that would not be possible using the macro.

You may also use the [`@addloglikelihood`](@ref) macro to increment the ... well, loglikelihood manually. Be aware that you sacrifice the ability to call `rand` on your model, and that many other parts of the Bayesian workflow will also not integrate as smoothly with models using [`@addloglikelihood`](@ref).

[^1]: See for example [MeasureBase.jl](https://github.com/JuliaMath/MeasureBase.jl). Wren also works with [Distributions.jl](https://github.com/juliastats/distributions.jl) but note that these distributions may not be GPU-compatible.

[^2]: A zero-inflated Poisson distribution accounts for zero counts that would not by described by a standard Poisson by mixing a Bernoulli and Poisson component.
