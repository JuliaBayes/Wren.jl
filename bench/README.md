# Wren benchmarks

This folder contains a couple of benchmarks of Wren vs Turing.

Note that a lot of this is vibe coded, it will hopefully be cleaned up in the future!
However it works more or less works as intended.

To run the benchmarks, run the following from the `bench` directory:

```sh
julia --project=. ./basic/run.jl
julia --project=. ./posteriordb/run.jl
```

`basic` compares three different models:

  - a trivial `x ~ Normal()` model;
  - eight-schools;
  - a multinomial logistic regression model on MNIST data.

`posteriordb` compares 10 PosteriorDB models.
