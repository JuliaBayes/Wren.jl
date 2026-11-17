Wren is a library for probabilistic programming in Julia.

## Installation

As Wren is not yet registered in the Julia package registry, you can install it directly from GitHub using:

```julia
import Pkg
Pkg.add(url="https://github.com/JuliaBayes/Wren.jl")
```

## Quick start

```@example index
using Wren, Wren.Dists

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(0, 5)
    theta ~ MvNormalScale(fill(mu, J), tau * I)
    @batch for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

model = eight_schools(y, sigma, J)
```

### Draw from the prior

```@example index
simulate(model)
```

### MCMC sampling

```@example index
import Enzyme
nuts(model, 1000, 4)
```
