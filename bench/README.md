# Wren benchmarks

## Benchmark suite

`suite/` benchmarks Wren's primal, Enzyme reverse, and Mooncake reverse on a set of
PosteriorDB models, and saves the results to a JSON file:

```sh
julia --project=bench bench/suite/run.jl results.json
```

On every push to `main`, CI runs this and saves the file to `runs/` on the
`benchmark-results` branch.

`web/` is a small website that shows one run as a table, or compares two runs (pick a
'tip' and a 'base'; cells are coloured by how much faster or slower tip is, and hovering
shows the base time). To run it locally:

```sh
cd bench/web
pnpm install
scripts/fetch-results.sh   # copy runs from the benchmark-results branch
pnpm dev
```

You can also load JSON files from disk with the 'Load file…' buttons.

## Other benchmarks

This folder contains a couple of benchmarks of Wren vs Turing.

Note that a lot of this is vibe coded, it will hopefully be cleaned up in the future!
However it works more or less works as intended.

To run the benchmarks, run the following from the `bench` directory:

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
