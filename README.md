# Wren.jl

[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://pysm.dev/Wren.jl/) <a href="https://pysm.dev/Wren.jl/"><img src="/docs/src/assets/logo.svg" align="right" alt="Wren.jl logo"></a>

Wren is a probabilistic programming language that harnesses modern Julia tools to maximise performance.
It relies on staged computation instead of directly emitting Julia code as, for example, [DynamicPPL](https://github.com/TuringLang/DynamicPPL.jl) does.
Wren also supports powerful syntax-level vectorisation tools via the `@batch` macro.

## Quick start

```julia
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

rand(model)
# VarNamedTuple
# ├─ mu => -4.868066951440143
# ├─ tau => 1.77728015115861
# └─ theta => [-5.645093320047867, -3.2642262056243085, -1.4107903497807133, -6.381125278012095, -7.83320633960707, -5.479364446358099, -5.0377262033159305, -4.311316591708384]

using Wren.Samplers, Enzyme
nuts(model, 1000, 3)
# ────────────────────────────────── progress ──────────────────────────────────
# Chain 1/3: found initial stepsize ● ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ● 1/1 100%
# Sampling (chain 1/3) ● ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ● 1000/1000 100%
# Chain 2/3: found initial stepsize ● ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ● 1/1 100%
# Chain 3/3: found initial stepsize ● ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ● 1/1 100%
# Sampling (chain 2/3) ● ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ● 1000/1000 100%
# Sampling (chain 3/3) ● ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ● 1000/1000 100%
# Sampling done in 0.13 seconds.
# Constructing chain... took 0.0 seconds
# ╭─FlexiChain (1000 iterations, 3 chains) ─────────────────────────────────────────╮
# │ ↓ iter  = 1:1000                                                                │
# │ → chain = 1:3                                                                   │
# │                                                                                 │
# │ Parameters (3) ── VarName                                                       │
# │  Float64          mu, tau                                                       │
# │  Vector{Float64}  theta (8,)                                                    │
# │                                                                                 │
# │ Extras (9)                                                                      │
# │  Float64  step_size, acceptance_rate, log_density, logprior, loglikelihood,     │
# │           logjoint                                                              │
# │  Int64    tree_depth                                                            │
# │  Bool     is_divergent, is_max_depth                                            │
# ╰─────────────────────────────────────────────────────────────────────────────────╯
```
