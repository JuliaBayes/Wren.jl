# Loading PosteriorDB models

Wren (technically, WrenCore) provides a PosteriorDB extension where some PosteriorDB models are available for use.
(Not all have been added yet, but the intent is that Wren will eventually provide a complete collection.)

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

You can then e.g. sample from this posterior:

```@example posteriordb
using Enzyme # Wren's default AD backend.

nuts(model, 1000, 4)
```

## Differences with Stan models

In general, the vast majority of PosteriorDB implementations in Wren have been very faithfully translated from Stan.
There are a few differences which we note here:

 1. Stochastic generated quantities haven't been included in the model (for example things like `generated quantities { x = binomial_rng(n, p); }`).
    We probably want to implement these in Wren properly: see [issue #330](https://github.com/JuliaBayes/Wren.jl/issues/330).

    Where possible we have already included deterministic generated quantities, as well as transformed parameters, as `@track` nodes in the Wren model.
    In particular, for transformed parameters, this matches Stan's behaviour by including them in the chain.
    Note that for benchmarking purposes these tracked quantities will not contribute to performance as log-density evaluations are always performed with tracking off.

 2. Some constant log-density terms have been added / dropped, for example, where Stan uses a flat constrained prior Wren sometimes uses `Uniform`, which adds a constant log-density of `1 / (upper - lower)` to the log-density.

 3. `Mh` is somewhat rewritten in order to make use of the batching in Wren; it has the same log-density but is expressed quite differently.

 4. In the `Rates` models we replace `Beta(1, 1)` with `Uniform(0, 1)` which has a cheaper log-density evaluation.

## Docstrings

```@docs; canonical=false
available_posteriordb_models
posteriordb_model
```
