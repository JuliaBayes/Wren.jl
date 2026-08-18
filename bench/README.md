# Wren vs DynamicPPL AD benchmarks

This environment compares linked-vector log-density and prepared-gradient evaluation for
the local Wren checkout's `VecModel` and the registered DynamicPPL release's
`LogDensityFunction`.

The benchmark matrix contains:

  - a single `x ~ Normal()` latent;
  - the eight-schools model from Wren's top-level README;
  - ForwardDiff;
  - ReverseDiff, both interpreted and compiled;
  - Mooncake;
  - Enzyme in reverse mode;
  - Reactant with forward- and reverse-mode Enzyme for Wren's
    `ProbabilityMeasures.Normal` model.

Wren passes model metadata to DifferentiationInterface as explicit constants and therefore
needs neither an Enzyme function annotation nor runtime activity. DynamicPPL receives
Enzyme's runtime-activity configuration, as required by its current AD path.
Unsupported model/backend combinations are reported as concise error rows and do not stop
the rest of the benchmark matrix.

Reactant is run only for the single-normal Wren model. Its DynamicPPL counterpart uses
Distributions.jl, which is not currently Reactant-compatible, so that row reports only
Wren timings and verifies its value and gradient against Wren's ordinary Enzyme backend.

The runner verifies that Wren and DynamicPPL return matching values and gradients at the
same parameter vector before benchmarking them.

From the Wren repository root:

```sh
julia --project=bench -e 'using Pkg; Pkg.instantiate()'
julia --project=bench bench/run.jl
```

Each Chairmarks measurement samples for one second by default. Set
`WREN_BENCH_SECONDS` to change this, for example:

```sh
WREN_BENCH_SECONDS=0.2 julia --project=bench bench/run.jl
```

The `speedup` columns report `DynamicPPL time / Wren time`, so values greater than one
favour Wren. Allocation columns are displayed as `DynamicPPL/Wren`.
