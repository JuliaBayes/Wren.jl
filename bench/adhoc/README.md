# Ad hoc benchmarks

This folder contains a couple of benchmarks of Wren vs Turing. It has its own environment,
separate from the main benchmark suite in `bench/`.

Note that a lot of this is vibe coded, it will hopefully be cleaned up in the future!
However it works more or less works as intended.

To run the benchmarks, run the following from the `bench/adhoc` directory:

```sh
julia --project=. ./basic/run.jl
julia --project=. ./posteriordb/run.jl
julia --project=. ./batch/run.jl
```

`basic` compares three different models:

  - a trivial `x ~ Normal()` model;
  - eight-schools;
  - a multinomial logistic regression model on MNIST data.

`posteriordb` compares 10 PosteriorDB models.

`batch` compares `@batch` loops against plain `for` loops for the likelihoods in
WrenDists that have specialised `batched` methods.

`batch/gpu` benchmarks alternative formulations of each batched log density, one script
per distribution, e.g. `julia --project=. ./batch/gpu/diagnormal.jl`. Each variant is
timed under plain Enzyme on the CPU and under Reactant on whatever device it picks. On a
machine with a GPU only the Reactant backend runs by default; set `WREN_BENCH_BACKENDS`
to `enzyme`, `reactant` or `enzyme,reactant` to override, and `WREN_BENCH_NS` to choose
the sizes.

`halfdists` compares `HalfNormal`, `HalfCauchy`, and `HalfTDist` with Distributions.jl
truncations. It checks and times construction, density, sampling, and Enzyme/Reactant
model densities and gradients. Run from `bench/adhoc` in a temporary environment:

```julia
using Pkg
Pkg.activate(; temp=true)
Pkg.develop([PackageSpec(path="../../WrenCore"), PackageSpec(path="../../WrenDists")])
Pkg.add(["ADTypes", "Chairmarks", "Distributions", "Enzyme", "ForwardDiff", "Reactant"])
include("halfdists/run.jl")
```

Set `WREN_BENCH_NS` (default `1,1000`), `WREN_BENCH_SECONDS` (`1.0`),
`WREN_BENCH_BACKENDS` (`enzyme,reactant`), and `WREN_BENCH_DEVICE` (`auto`, `cpu`, or
`gpu`; default `auto`) for Reactant. Models differentiate samples with fixed
distribution parameters. Reactant timings include host transfers and name the actual
device. Allocations measure host memory. `construct+density` creates one distribution
for all `N` observations.

`isonormal` compares three ways of evaluating a batched normal whose `sigma` is the same
for all elements: `IsoNormal`, using scalar `sigma`; `DiagNormal+Fill`, applying the
elementwise logdensity to a `Fill` array; and `BatchedNormal+Fill`. The first two types
are defined in the script because they were superseded by `BatchedNormal` in WrenDists.
The benchmark checks and times construction, density, sampling, and Enzyme/Reactant model
densities and gradients. Run from `bench/adhoc` in a temporary environment:

```julia
using Pkg
Pkg.activate(; temp=true)
Pkg.develop([PackageSpec(path="../../WrenCore"), PackageSpec(path="../../WrenDists")])
Pkg.add(["ADTypes", "Chairmarks", "Distributions", "Enzyme", "FillArrays", "Reactant"])
include("isonormal/run.jl")
```

Set `WREN_BENCH_NS` (default `10,1000,100000`), `WREN_BENCH_SECONDS` (`1.0`),
`WREN_BENCH_BACKENDS` (`enzyme,reactant`), and `WREN_BENCH_DEVICE` (`auto`, `cpu`, or
`gpu`; default `auto`) for Reactant. The `vs IsoNormal` column divides each median time by
the `IsoNormal` median of the same operation. `IsoNormal (repeat)` times `IsoNormal` a
second time, so its ratio shows the drift between two measurements of the same code.
Reactant timings include host transfers. Allocations measure host memory.
