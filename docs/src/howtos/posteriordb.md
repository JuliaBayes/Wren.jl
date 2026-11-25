# [PosteriorDB models](@id posteriordb)

Wren provides a PosteriorDB extension where some PosteriorDB models are available for use.
(Not all have been added yet, but the intent is that Wren will eventually provide a complete collection.)

## Loading models

To list the available models, use:

```@example posteriordb
using Wren, PosteriorDB

available_posteriordb_models()
```

!!! note "Benchmarks"

    Some of these PosteriorDB models are also benchmarked in order to track performance over time: see [https://juliabayes.org/Wren.jl/bench](https://juliabayes.org/Wren.jl/bench).

And to load one of these:

```@example posteriordb
model = posteriordb_model("eight_schools-eight_schools_centered")
```

You can then e.g. sample with it:

```@example posteriordb
using Enzyme # Wren's default AD backend.

nuts(model, 1000, 4)
```

## eference posteriors

PosteriorDB.jl also provides reference posteriors, i.e., chains sampled in Stan, for several of the models.
FlexiChains contains an extension which lets you load these as `VNChain` objects.

This is not part of Wren itself.
However, since FlexiChains is a dependency of Wren, you will automatically be able to access these when importing Wren and PosteriorDB, so we mention it here for completeness.

```@example posteriordb
using FlexiChains, PosteriorDB

# Show available reference posteriors
FlexiChains.available_posteriordb_references()
```

```@example posteriordb
# Load a reference posterior
FlexiChains.from_posteriordb("eight_schools-eight_schools_centered")
```

Please see [the FlexiChains docs](@extref FlexiChains integrations-posteriordb) for more info.

## Model source code

If you want to see the Wren implementations, here is [a handy link to the relevant directory on GitHub](https://github.com/JuliaBayes/Wren.jl/tree/main/WrenCore/ext/WrenCorePosteriorDBExt/models).

## Differences with Stan models

In general, the vast majority of PosteriorDB implementations in Wren have been very faithfully translated from Stan.
There are a few differences which we note here:

 1. Stochastic generated quantities haven't been included in the model (for example things like `generated quantities { x = binomial_rng(n, p); }`).
    We probably want to implement these in Wren properly: see [issue #330](https://github.com/JuliaBayes/Wren.jl/issues/330).

    Where possible we have already included deterministic generated quantities, as well as transformed parameters, as `@track` nodes in the Wren model.
    In particular, for transformed parameters, this matches Stan's behaviour by including them in the chain.
    Note that for benchmarking purposes these tracked quantities will never contribute to performance as log-density evaluations are always performed with tracking off.

 2. Some constant log-density terms have been added / dropped.
    For example, where Stan uses a flat constrained prior Wren sometimes uses `Uniform`, which adds a constant log-density of `-log(upper - lower)`.
    Wren's `HalfNormal` and `HalfCauchy` distributions also include a constant normalisation term which Stan drops.

 3. `Mh` is somewhat rewritten in order to make use of the batching in Wren; it has the same log-density but is expressed quite differently.

 4. In the `Rates` models we replace `Beta(1, 1)` with `Uniform(0, 1)` which is the same distribution but has a cheaper log-density evaluation.

## Docstrings

```@docs; canonical=false
available_posteriordb_models
posteriordb_model
```
