# Wren.jl

[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliabayes.org/Wren.jl/)

Probabilistic programming with runtime generated functions.

```julia
using Wren
using Distributions
using LinearAlgebra: I

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8
@model function eight_schools(y, sigma, J)
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
```

## Benchmarks

Go to `bench/` and run `julia +1.13 run.jl`:

```
model           backend            DPPL lp     Wren lp   speedup   lp allocs   DPPL grad   Wren grad   speedup     grad allocs
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
normal          ForwardDiff       2.580 ns    2.992 ns     0.86x         0/0   16.889 ns   15.501 ns     1.09x             3/3
normal          ReverseDiff       2.677 ns    3.614 ns     0.74x         0/0    1.935 μs    1.958 μs     0.99x           64/63
normal          ReverseDiffCpl    2.851 ns    4.050 ns     0.70x         0/0  758.538 ns  729.175 ns     1.04x             3/3
normal          MooncakeFwd       2.717 ns    3.874 ns     0.70x         0/0  114.629 ns   85.985 ns     1.33x           10/10
normal          MooncakeRvs       2.738 ns    4.043 ns     0.68x         0/0   73.677 ns   36.966 ns     1.99x             2/2
normal          DifferFwd         2.513 ns    2.844 ns     0.88x         0/0   31.382 ns   35.405 ns     0.89x             6/6
normal          DifferRvs         2.600 ns    2.988 ns     0.87x         0/0   38.046 ns   37.197 ns     1.02x             2/2
normal          EnzymeFwd         2.575 ns    2.983 ns     0.86x         0/0   86.111 ns   87.009 ns     0.99x             5/5
normal          EnzymeRvs         2.521 ns    2.848 ns     0.89x         0/0   11.387 ns   11.786 ns     0.97x             2/2
normal          ReactantRvs              —    9.375 μs         —        —/21           —   15.959 μs         —            —/27
eight schools   ForwardDiff      75.867 ns   62.063 ns     1.22x         4/4  358.466 ns  348.129 ns     1.03x           11/11
eight schools   ReverseDiff      75.714 ns   61.850 ns     1.22x         4/4   20.458 μs   21.000 μs     0.97x         596/592
eight schools   ReverseDiffCpl   74.773 ns   64.381 ns     1.16x         4/4    6.521 μs    6.292 μs     1.04x             6/4
eight schools   MooncakeFwd      76.951 ns   63.278 ns     1.22x         4/4    3.429 μs    2.579 μs     1.33x         148/148
eight schools   MooncakeRvs      77.247 ns   63.968 ns     1.21x         4/4  689.103 ns  524.860 ns     1.31x           12/12
eight schools   DifferFwd        76.110 ns   62.046 ns     1.23x         4/4    2.891 μs    2.542 μs     1.14x         168/168
eight schools   DifferRvs       ERROR: Differ.build_ctx: could not derive a reverse rule for `Tuple{typeof(Wren._vec...
eight schools   EnzymeFwd        76.572 ns   63.743 ns     1.20x         4/4  648.513 ns  566.295 ns     1.15x           49/49
eight schools   EnzymeRvs        76.328 ns   62.821 ns     1.21x         4/4  232.143 ns  163.745 ns     1.42x           13/11
mnist           ForwardDiff      68.666 μs   67.041 μs     1.02x       28/26   15.778 ms   16.147 ms     0.98x         872/810
mnist           ReverseDiff     ERROR: MethodError: vcat(::Matrix{ReverseDiff.TrackedReal{Float64, Float64, ReverseD...
mnist           ReverseDiffCpl  ERROR: MethodError: vcat(::Matrix{ReverseDiff.TrackedReal{Float64, Float64, ReverseD...
mnist           MooncakeFwd      68.792 μs   66.959 μs     1.03x       28/26  241.755 ms  238.317 ms     1.01x 4827661/4826183
mnist           MooncakeRvs      68.291 μs   66.750 μs     1.02x       28/26  684.083 μs  678.500 μs     1.01x         179/175
mnist           EnzymeFwd        68.416 μs   67.084 μs     1.02x       28/26   65.227 ms   69.347 ms     0.94x     14063/13229
mnist           EnzymeRvs        67.708 μs   66.417 μs     1.02x       28/26  366.875 μs  257.209 μs     1.43x          125/83
mnist           ReactantRvs              —   70.584 μs         —        —/21           —  105.333 μs         —            —/28
```

### Syntax

#### Model arguments

In DynamicPPL, any identifier passed as an argument to the model is treated as an observation:

```julia
@model function f(y)
    x ~ Normal()
    y ~ Normal(x, 1) # DynamicPPL sees this as an observation
end
```

With Wren, all identifiers on the LHS of tilde-statements are treated as latent variables, unless they are conditioned on, or marked with `@observe` (see below).
Thus, in Wren you can either write

```julia
@model function f()
    x ~ Normal()
    y ~ Normal(x, 1) # Wren sees this as a latent
end
model = f() | VarNamedTuple(; y)
```

or

```julia
@model function f(y)
    x ~ Normal()
    @observe y ~ Normal(x, 1)
end
```

This avoids DynamicPPL's somewhat confusing dual interface where you can either condition or pass in an argument (the latter behaviour has mostly been kept around just for backwards compatibility).
The semantics in Wren also helps navigate awkward situations such as

```julia
@model function f2(y)
    z = y + 1
    z ~ Normal()
end
```

With DynamicPPL, `z` is derived from data and thus one might expect it to be treated as an observation, but DynamicPPL can't detect this and thus treats it as a latent.
With Wren there is simply no question: `z` is a latent.

#### `@observe`

As introduced above, Wren provides an `@observe` macro to indicate that a variable should always be treated as an observation:

```julia
@model function f()
    x ~ Normal()
    y = x + 1
    @observe y ~ Normal()
end
```

In DynamicPPL, there is no real way to accomplish this.
You can manually add the log-likelihood with `@addlogprob! logpdf(Normal(), y)`, but this doesn't create a sampling site for `y` (i.e., `y` is not recognised as a variable in the model).
That means that you can't, for example, extract a pointwise log-likelihood for `y`.

See [DynamicPPL#1270](https://github.com/TuringLang/DynamicPPL.jl/issues/1270) for the discussion where I originally came up with this idea.
