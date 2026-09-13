# Wren benchmarks

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

`batch/gpu` benchmarks alternative formulations of each batched log density under
Reactant, one script per distribution, e.g. `julia --project=. ./batch/gpu/diagnormal.jl`.
It runs on whatever device Reactant picks, so it can be smoke-tested on a CPU and then
run for real on a GPU.
