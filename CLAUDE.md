# Instructions for Agents

- **Always** use a temporary environment with Wren.jl dev'd into it to run code.
  This avoids adding deps into Wren itself. For example, Distributions is not a
  dependency of Wren.

- `DensityInterface.logdensityof`, most relevant ADTypes, and `Enzyme.Reverse` are
  re-exported by Wren. No need to import them.

# Running models

```
using Wren, Distributions

@wren function f(a)
    x ~ Normal(a)
end

model = f(1.0)
```

- `rand(model)` draws a sample from the model.
- `rand_with_logp(model)` draws a sample plus the components of log density.

For more information about model syntax, see `docs/src/syntax.md`.

# Vectorised models

```
vecmodel = VecModel(model, AutoEnzyme(; mode=Reverse)) # or any other adtype
v = rand(vecmodel) # draws a vector
```

- `lp(vecmodel, v)` computes the log density.
- `lp_and_grad(vecmodel, v)` computes a (logdensity, gradient) tuple.

# Benchmarking models

Basic, direct usage:

```
using Chairmarks

@b rand($model)
@b lp($vecmodel, $v)
```

Always make sure to interpolate the arguments to avoid spurious allocations.
