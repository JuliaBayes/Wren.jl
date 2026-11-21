# MCMC sampling

Wren provides prior sampling, Metropolis–Hastings, and NUTS.

Results are stored as `FlexiChain{VarName}`; please see the [FlexiChains documentation](https://juliabayes.org/FlexiChains.jl) for more information.

```@example mcmc
using Wren, Wren.Dists

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(0, 5)
    theta ~ MvNormalScale(fill(mu, J), tau * I)
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

## [Metropolis–Hastings](@id metropolis-hastings)

Independent MH proposes from the prior. The random-walk samplers use Gaussian proposals in
unconstrained space. RAM, AM, and DRAM adapt the proposal covariance during warmup.

```@docs
Wren.independent_mh
Wren.rwm
Wren.ram
Wren.am
Wren.dram
```

```@example mcmc
ram(model, 1000, 4; progress=false)
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
