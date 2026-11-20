# MCMC sampling

Wren provides prior sampling, NUTS and optional ensemble MCMC.

Results are stored as `FlexiChain{VarName}`; please see the [FlexiChains documentation](https://juliabayes.org/FlexiChains.jl) for more information.

```@example mcmc
using Wren
using Distributions
using LinearAlgebra: I

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ MvNormal(fill(mu, J), tau^2 * I)
    for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

model = eight_schools(y, sigma, J)
```

## Prior

```@docs
Wren.prior
```

```@example mcmc
using Wren
prior(model, 1000, 4; progress=false)
```

## [NUTS](@id nuts)

```@docs
Wren.nuts
```

```@example mcmc
using Enzyme
adtype = AutoEnzyme(; mode=Reverse)

using Wren
nuts(model, 1000, 4; adtype=adtype, progress=false)
```

## Ensemble MCMC

Ensemble MCMC needs no gradients. Load EnsembleMCMC, then pass the model directly:

```@example mcmc
using EnsembleMCMC
walkers = ensemble(model, 1000, 4; warmup=500, progress=false);
walkers[1]
```

Starting positions are drawn automatically. Each returned chain tracks one walker
across the four independent ensembles. Use these chains separately for diagnostics;
walkers within an ensemble are dependent. See [`ensemble`](@ref) for custom starts and moves.

GPU sampling uses Reactant, CUDA and compatible distributions from `Wren.Dists`:

```julia
using Reactant, CUDA
using ADTypes: AutoReactant

@wren function rate_model(count)
    rate ~ Wren.Dists.Exponential()
    @observe count ~ Wren.Dists.Poisson(rate)
end

Reactant.set_default_backend("gpu")
walkers = ensemble(
    rate_model(3),
    1000;
    adtype=AutoReactant(AutoEnzyme(; mode=Reverse)),
    progress=false,
)
```

Model evaluations, ensemble moves and retained history run on Reactant's selected
GPU. The sampler returns the completed chains on the CPU.
