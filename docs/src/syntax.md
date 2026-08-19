# [Syntax and semantics](@id syntax)

A Wren model is a Julia function marked with the `@model` macro.
It mixes regular Julia code with two probabilistic primitives:

  - `sample`: represents latent parameters
  - `observe`: represents observations

Function arguments have no special meaning in Wren: they're just quantities that are passed into the model.

## `sample` with distributions

**All** `x ~ y` statements are lowered into `sample` statements.

The simplest case is where you have a random variable drawn from a distribution:

```@example model
using Wren, Distributions

@model function f()
    x ~ Normal()
end

model = f()
```

This model has a single latent variable, `x`, drawn from `Normal()`.

Wren uses a concept of *accumulators* to represent outputs from a model.
We'll use a few of Wren's functions to extract information from these; we won't explain them here, but please do see [Models](@ref models) for more usage details.

```@example model
accs = rand_with_logp(model)
logprior(accs), loglikelihood(accs), params(accs)
```

Here we can see that a value was sampled for `x`, and that the log-prior (and not
log-likelihood) was updated.

## Arrays and structs

The left-hand side of a tilde can be either

  - a top-level variable, e.g., `x ~ Normal()`,
  - a field, e.g., `m.x ~ Normal()`,
  - an array element, e.g., `a[1] ~ Normal()`,
  - or chains thereof, e.g., `m.a[1].x ~ Normal()`.

These are all things which can be assigned to in Julia (sometimes called 'lvalues').
Other expressions are not permitted, including literals.

Note that if you have a field or element on the left-hand side, the parent object must already exist.
That is, you cannot write `m.x ~ Normal()` if `m` is not already defined.

For example, this is permitted since `y` is an argument to the model function (assuming that `y` is something that can be indexed into):

```@example model
using Random

@model function yarg(y)
    x ~ Normal()
    for i in eachindex(y)
        y[i] ~ Normal(x)
    end
end
my_y = zeros(3)
yarg_model = yarg(my_y)

rand(Xoshiro(468), yarg_model)
```

And this is permitted because `y` is defined in the model body:

```@example model
@model function ydef()
    x ~ Normal()
    y = zeros(3)
    for i in eachindex(y)
        y[i] ~ Normal(x)
    end
end
ydef_model = ydef()

rand(Xoshiro(468), ydef_model)
```

It should be noted that the two models above are actually semantically equivalent: in both models, all `y[i]` are latent variables, regardless of whether `y` is an argument or defined in the model body.

The first version (`yarg`) is more efficient, as it avoids performing the allocation of `y` each time the model is run.

```@example model
using Chairmarks
@b rand($yarg_model)
```

```@example model
@b rand($ydef_model)
```

Be aware though that the argument to the model will be mutated!
Thus, one can really think of the function argument as **a 'scratch' buffer that declares the shape of the latent array**, whose contents are not used in any way.

```@example model
my_y
```

Immutable structs largely behave the same way as top-level variables, and mutable structs follow the same principles as arrays.

## Conditioning

Now, we would almost certainly like to progress from the starting point above where everything is a latent variable.
To condition on some observed data, we can use the `|` operator on the model.
For example, we can condition on the entire `y` array in one go:

```@example model
yarg_model_cond = yarg_model | VarNamedTuple(y=[1.0, 2.0, 3.0])

accs = rand_with_logp(Xoshiro(468), yarg_model_cond)
logprior(accs), loglikelihood(accs), params(accs)
```

Now the only latent variable is `x`, and the `y` array is observed.

Note that the values in the argument passed to the model are now no longer used!
As before, the `y` model argument only serves to declare the shape of the latent array.
When the model is evaluated, it will use the values passed to `|` instead of the values in the argument.

We can do the exact same with `ydef`, and in exactly the same way, the definition of `y` inside the model body will be ignored:

```@example model
ydef_model_cond = ydef_model | VarNamedTuple(y=[1.0, 2.0, 3.0])

accs = rand_with_logp(Xoshiro(468), ydef_model_cond)
logprior(accs), loglikelihood(accs), params(accs)
```

One can perform partial conditioning by only specifying part of the latent array (see the [VarNames.jl](https://juliabayes.org/VarNames.jl/stable/vnt/api/#VarNames.@vnt) docs for more details on the `@vnt` macro):

```@example model
cond = @vnt begin
    @template y = zeros(3)
    y[1] := 1.0
    y[3] := 3.0
end

yarg_model_partial_cond = yarg_model | cond
rand(yarg_model_partial_cond)
```

## Enforced observations

Sometimes you may want to enforce in a model that a quantity is observed.
However, you might not be able to provide this information at the time of model definition (for example, this might depend on the value of another variable).

Take for example:

```@example model
@model function y_not_obs()
    x ~ Normal()
    y = x + 1
    y ~ Normal()
end
```

In this model, even though `y` is defined in the model, the subsequent statement `y ~ Normal()` will treat `y` as a latent variable, and will sample a new value for it, which will certainly not be equal to `x + 1`:

```@example model
rand_with_logp(y_not_obs())
```

To force `y` to be observed, Wren provides a macro `@observe`:

```@example model
@model function y_obs()
    x ~ Normal()
    y = x + 1
    @observe y ~ Normal()
end

accs = rand_with_logp(y_obs())
```

We can check that these values are consistent:

```@example model
loglikelihood(accs) ≈ logdensityof(Normal(), params(accs)[@vn(x)] + 1)
```

You can in principle use this to enforce that model arguments are observed:

```@example model
@model function yarg_obs(y)
    x ~ Normal()
    for i in eachindex(y)
        @observe y[i] ~ Normal(x)
    end
end

rand(yarg_obs([1.0, 2.0, 3.0]))
```

However, note that once a statement has been marked with `@observe`, you cannot 'uncondition' it without modifying the model's stored AST itself.
