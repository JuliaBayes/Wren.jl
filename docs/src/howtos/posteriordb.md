# Loading PosteriorDB models

Wren (technically, WrenCore) provides a PosteriorDB extension where some PosteriorDB models are available for use.
(Not all have been added yet, but the intent is that Wren will eventually provide a complete collection.)

To list the available models, use:

```@example posteriordb
using Wren, PosteriorDB

available_posteriordb_models()
```

!!! note "Benchmarks"

    These PosteriorDB models are also benchmarked in order to track performance over time: see [https://pysm.dev/Wren.jl/bench](https://pysm.dev/Wren.jl/bench).

And to load one of these:

```@example posteriordb
model = posteriordb_model("eight_schools-eight_schools_centered")
```

You can then e.g. sample from this posterior:

```@example posteriordb
using Enzyme # Wren's default AD backend.

nuts(model, 1000, 4)
```

## Docstrings

```@docs
available_posteriordb_models
posteriordb_model
```
