# Wren.jl

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

## Just tell me whether it's better

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

## What's taken from DynamicPPL

Much of the code in Wren is taken from [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl), which my colleagues and I developed for two years.
In particular, DynamicPPL's entire framework of initialisation strategies and accumulators has been copied wholesale, along with the `VarName` and `VarNamedTuple` types (the latter have been extracted into a separate package, [VarNames.jl](https://github.com/JuliaBayes/VarNames.jl).
There is a small change: what DynamicPPL calls 'initialisation strategies', Wren calls 'latent sources' instead, which is clearer (the 'initialisation' term in DynamicPPL is mostly a historical artifact, which was chosen in order to try to differentiate it from `DynamicPPL.DefaultContext`).

The reason for this is because this design is very flexible, and accommodates most of the things one would want to do with a probabilistic model.
`VarName` and `VarNamedTuple` are, as far as I know, the best ways of representing variable addresses in a probabilistic programme, both in terms of generality as well as performance.
Model evaluation has three explicit stages: a latent source supplies a value and its current representation, a parameter space selects the coordinate space and computes the forward log-Jacobian, and accumulators collect the resulting information.
This design is also essentially optimal in terms of performance: for many models, its overhead is close to zero.
All the improvements in Turing's performance in recent years have been due to these design choices.
See, for example, these PRs: [DynamicPPL#1125](https://github.com/TuringLang/DynamicPPL.jl/pull/1125), [DynamicPPL#1132](https://github.com/TuringLang/DynamicPPL.jl/pull/1132), [DynamicPPL#1183](https://github.com/TuringLang/DynamicPPL.jl/pull/1183), [DynamicPPL#1211](https://github.com/TuringLang/DynamicPPL.jl/pull/1211).

## Differences WRT DynamicPPL

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

#### Submodels

Wren separates the return value of a submodel from the namespace within which it is evaluated.
To illustrate this, in the following,

```julia
julia> rand(f())
@model function inner()
    x ~ Normal()
    return 42
end

@model function f()
    retval = a ~ inner()
    @show retval
end
```

the result will have a single key `a.x` (`a` being the prefix), and `retval` will be assigned the value 42.

DynamicPPL conflates these two concepts, i.e., `a ~ inner()` samples `inner`'s variables with the prefix `a`, but also assigns `inner`'s return value to `a`.
This leads to a lot of ambiguity.
For example, in `a ~ inner()`, since `a` represents the return value, you can't condition on `a`, even though it _looks_ as though you can (since it's on the left-hand side of a tilde).
In contrast with Wren the syntax makes it clear that you can condition on `a` (or any of its subvariables), but not `retval`.

This has been a long-standing issue in DynamicPPL (see [Turing#2485](https://github.com/TuringLang/Turing.jl/issues/2485), and also [Turing's docs page on submodels](https://turinglang.org/docs/usage/submodels/#submodels-versus-distributions), which I wrote).

### DensityInterface.jl

Wren uses `DensityInterface.logdensityof` rather than `Distributions.logpdf`.
Indeed, Wren does not even have a dependency on a distribution library.

The main intention here is to allow users to pass in whatever distributions they want.
In particular, a medium-term goal in Wren is to pass vectorised model evaluation functions (i.e., the equivalent of `DynamicPPL.LogDensityFunction`) to [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), and Distributions.jl is very much incompatible with Reactant.

### Architecture

#### The model macro itself

DynamicPPL's model macro is a very thin transformation layer.
It essentially converts `x ~ dist` into `tilde_assume!!(x, dist, ...)` or `tilde_observe!!(x, dist, ...)`, and uses multiple dispatch to imbue this with probabilistic semantics.
For example, if you run `@macroexpand DynamicPPL.@model f() = x ~ Normal()`, you will get something that (once you strip away all the bells and whistles) looks similar to

```julia
function f(varinfo, context)
    DynamicPPL.tilde_assume!!(varinfo, context, :x, Normal())
end
function f()
    return DynamicPPL.Model(f, DefaultContext())
end
```

which is directly eval'd into wherever you ran the macro from.
Although this is a successful design, it suffers from limitations.
In particular, once you create a DynamicPPL model, it is opaque: you can't inspect its contents to see that there is a single variable `x` that follows a `Normal` distribution.
Indeed, the only way you can figure this out is by actually evaluating the model.

Wren switches this up such that `f()` instantiates a model that contains its own AST, something that looks like:

```julia
Model(:f, [Sample(:x, :(Normal()))])
```

Some transformations have been performed here (for example `lhs ~ rhs` is transformed into `Sample(lhs, rhs)`).
The evaluation function itself, which is analogous to the `f(varinfo, context)` method that DynamicPPL generates, is generated at runtime instead of at macro expansion time.
This is accomplished using [RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl).
The present approach is inspired by [Soss.jl](https://github.com/cscherrer/Soss.jl) and its successor [Tilde.jl](https://github.com/cscherrer/Tilde.jl), which do something similar with [GeneralizedGenerated.jl](https://github.com/JuliaStaging/GeneralizedGenerated.jl) instead of RGF.
GG is supposedly more powerful than RGF, but I haven't yet come across a use case where GG was necessary.
Furthermore, RGF is actually maintained (for example it is used in ModelingToolkit.jl), and so it seems like a better choice for a dependency.

The main, immediate, benefit of this design is that it lets you surgically modify the AST stored inside a `Model` object (and then regenerate the evaluation function).
For example, one could easily implement a transformation such as:

```julia
@model function f()
    x ~ Normal()
end

model = f()
new_model = set_distribution(model, :x, Beta(2, 2))
```

which would replace the original `Normal()` distribution with a `Beta(2, 2)` distribution.

Furthermore, storing the AST should in principle also allow for some level of static analysis of the model, which could be used to construct (for example) a graphical representation of the model.
(See, e.g., [GraphPPL.jl](https://github.com/ReactiveBayes/GraphPPL.jl) for an existing implementation of this.)
This would open up the possibility of more advanced inference algorithms which make use of this graphical representation, such as message passing.

#### Evaluation function arguments

Wren uses the equivalent of DynamicPPL's `OnlyAccsVarInfo` type, and refuses to include a full `VarInfo`.
DynamicPPL in fact does not need `VarInfo`: it can be quite easily removed from DynamicPPL (see [DynamicPPL#1376](https://github.com/TuringLang/DynamicPPL.jl/issues/1376)).
The only possible reason to keep it is for backwards compatibility.

Furthermore, DynamicPPL has a notion of a _context_, which `tilde_assume!!` and `tilde_observe!!` dispatch on.
Over the previous years, these contexts have been pared down massively.
However, in practice, `DynamicPPL.InitContext` is the only 'real' context that is meaningful (apart from Turing's particle MCMC sampler context; but that's a problem for the future).
The other major DynamicPPL context is `DefaultContext`, which _really_ means the same thing as `InitContext` with the parameters supplied from a `VarInfo` struct.
I have long argued that this is not necessary to keep around (see [DynamicPPL#1184](https://github.com/TuringLang/DynamicPPL.jl/issues/1184)), but it hasn't yet happened.

In essence I have opted to liberate Wren from the old and somewhat suboptimal design choices of DynamicPPL.

#### Submodel handling

DynamicPPL also uses `PrefixContext` to handle submodels.
This is a context that more or less says 'evaluate the submodel, but every time you see `x ~ dist`, treat it as `prefix_x ~ dist`'.

This happens to work decently for a lot of cases.
However, it's also simply not good software engineering!
Instead of forcing the submodel to be prefix-aware, it should be the responsibility of the caller to prefix the submodel's variables.

This is accomplished in Wren using the `scope` and `combine_with_namespace` functions.
The former is used to control the latent source (i.e., initialisation strategy) passed to the submodel, and the latter is used to combine the accumulators from submodel evaluation with the accumulators from the parent model evaluation.

I proposed this design a while back in [DynamicPPL#1221](https://github.com/TuringLang/DynamicPPL.jl/issues/1221), but it has never been implemented there.

Apart from being simply a better design, this also fixes some edge cases where top-level templates in the caller model were not used correctly when storing results from the submodel.
The issue above has an example of this.
