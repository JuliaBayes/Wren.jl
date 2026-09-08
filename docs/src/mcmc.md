# MCMC sampling

Wren currently ships with two very simple samplers.

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

## NUTS

```@docs
Wren.nuts
```

```@example mcmc
using Enzyme
adtype = AutoEnzyme(; mode=Reverse)

using Wren
nuts(model, 1000, 4; adtype=adtype, progress=false)
```
