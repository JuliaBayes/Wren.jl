# [Benchmarking](@id benchmarking)

The easiest way to benchmark models is to use the [`@bmodel`](@ref) macro:

```@example 1
using Wren, Wren.Dists

@wren function f()
    x ~ Exponential()
    z ~ MvNormalScale(zeros(3), abs(x) * I)
end

@bmodel f()
```

When passed a single argument, `@bmodel` will benchmark:

  - `rand`: the time it takes to sample a set of parameters from the model
  - `logjoint`: the time it takes to compute the log joint probability of a set of parameters
  - `lp`: the time it takes to compute the log-joint probability, but with [a vectorised interface](@ref vectorisation) (this is the same as e.g. `LogDensityProblems.logdensity`, if you are familiar with that interface).

To access the data in this benchmark, you can directly access the fields of the [`ModelRunResult`](@ref) object.

## AD

To additionally benchmark gradients, you can pass an `AbstractADType` as the second argument to `@bmodel`:

```@example 1
using Enzyme

# `Wren.DEFAULT_AD` is the same as `AutoEnzyme(; mode=Reverse)`.
@bmodel f() DEFAULT_AD
```

In addition to the three columns above, this will also benchmark `lp_and_grad`, which is the time taken to compute the log-joint plus its gradient (i.e., `LogDensityProblems.logdensity_and_gradient`).

## More control

If this isn't enough and you want to customise _exactly_ what gets benchmarked, and to test correctness, Wren also provides a function [`benchmark_and_test`](@ref) which has a staggering number of options.
(There are also slightly simpler versions, [`benchmark_ad`](@ref), and [`test_ad`](@ref).)
