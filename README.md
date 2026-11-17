# Wren.jl

[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliabayes.org/Wren.jl/) <a href="https://juliabayes.org/Wren.jl/"><img src="/docs/src/assets/logo.svg" align="right" alt="Wren.jl logo" style="height: 140px;"></a>

Wren is a library for defining and evaluating probabilistic models through staged compilation.
It is designed with performance on large models in mind, with a `@batch` macro for syntax-level vectorisation, as well as first-class support for Enzyme and Reactant.

## Quick start

```julia
using Wren, Wren.Dists

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

# Define a model
@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(0, 5)
    theta ~ MvNormalScale(fill(mu, J), tau * I)
    @batch for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu, tau)
end
model = eight_schools(y, sigma, J)

# Draw from the prior
simulate(model)

# MCMC sampling
import Enzyme
nuts(model, 1000, 4)
╭─FlexiChain (1000 iterations, 4 chains) ──────────────────────────╮
│ ↓ iter  = 1:1000                                                 │
│ → chain = 1:4                                                    │
│                                                                  │
│ Parameters (3) ── VarName                                        │
│  Float64          mu, tau                                        │
│  Vector{Float64}  theta (8,)                                     │
│                                                                  │
│ Extras (10)                                                      │
│  Float64                  step_size, acceptance_rate             │
│                           log_density, logprior, loglikelihood   │
│                           logjoint                               │
│  Int64                    tree_depth                             │
│  Bool                     is_divergent, is_max_depth             │
│  Tuple{Float64, Float64}  return_value                           │
╰──────────────────────────────────────────────────────────────────╯
```
