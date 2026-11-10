# [Syntax and semantics](@id syntax)

A Wren model is a Julia function marked with the `@wren` macro.
It mixes regular Julia code with two probabilistic primitives:

  - `sample`: represents latent parameters
  - `observe`: represents observations

Function arguments have no special meaning in Wren: they're just quantities that are passed into the model.

## `sample` with distributions

**All** `x ~ y` statements are lowered into `sample` statements.

The simplest case is where you have a random variable drawn from a distribution:

```@example model
using Wren, Distributions

@wren function f()
    x ~ Normal()
end

model = f()
```

This model has a single latent variable, `x`, drawn from `Normal()`.

Wren uses a `ModelResult` type to represent all the information that can be obtained by evaluating a model.
We'll use a few of Wren's functions to extract information from these.

```@example model
result = simulate(model)
logprior(result), loglikelihood(result), parameters(result)
```

Here we can see that a value was sampled for `x`, and that the log-prior (and not log-likelihood) was updated.

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

@wren function yarg(y)
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
@wren function ydef()
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

```@example model
@bmodel yarg_model
```

```@example model
@bmodel ydef_model
```

Immutable structs largely behave the same way as top-level variables, and mutable structs follow the same principles as arrays.

!!! note "No side effects"

    Left-hand sides of tilde-statements should not have side effects.
    If they do, Wren makes no guarantee as to what will happen!
    For example, `y[i + 1] ~ Normal()` is fine, but `y[i += 1] ~ Normal()` is not (it will parse but is undefined behaviour).

## Mutability

Related to the point above is that arguments which are mutable (e.g. arrays, Dicts, mutable structs) are not guaranteed to be copied upon every model evaluation.
This is done for performance reasons: for example, it means that you can preallocate a scratch array and pass it to the model so that it can be reused on every evaluation!
However, it does also mean that if you mutate an argument inside a model, this _may_ be visible on subsequent evaluations.
This is especially so if you make use of Wren's copy-elision optimisations.
(TODO: This doesn't exist yet, but it will soon.)

It is quite hard to _accidentally_ trigger such behaviour in an observable manner, and if you do so you will probably be able to guess that you are doing something wrong.
Here is a rather contrived example:

```@example model
@wren function mutable(y)
    # This reads the value of `y[1]`.
    @observe y[1] ~ Normal()
    # This mutates `y[1]` by sampling a new value for it.
    y[1] ~ Normal()
end

y_outside = [0.0]
model = mutable(y_outside)
model.args.y
```

Evaluating the model will now mutate `y`:

```@example model
rand(model);
model.args.y
```

and if we evaluate it again, the log-density associated with the `@observe` statement will change.

However, Wren guarantees that the value of `y_outside` will never be mutated.
This is because when Wren _constructs_ a model (i.e., when calling `mutable(y_outside)`), it will copy the value of `y_outside`.

```@example model
y_outside
```

!!! warning "Don't mutate latent variables or conditioned values"

    The guarantee above only applies to model arguments.
    Wren does **not** copy the values of latent variables, nor the values you condition on (see the next section).
    For performance, the value a model receives for a latent variable may be a view into the caller's parameter vector (for example, the vector passed to [`lp`](@ref), or a sampler's current position), and a conditioned value is the same object that you passed to `|`.

    So, writing into one of these inside a model, e.g.

    ```julia
    x ~ MvNormal(zeros(2), I)
    x[1] += 1.0
    ```

    is not supported: it can change the caller's data, make repeated evaluations return different log-densities, and break samplers.
    If you need a modified version, make a copy first (e.g. `x2 = copy(x); x2[1] += 1.0`), or write it without mutation (e.g. `x2 = x .+ [1.0, 0.0]`).

## Conditioning

Now, we would almost certainly like to progress from the starting point above where everything is a latent variable.
To condition on some observed data, we can use the `|` operator on the model.
For example, we can condition on the entire `y` array in one go:

```@example model
yarg_model_cond = yarg_model | (; y=[1.0, 2.0, 3.0])

res = simulate(Xoshiro(468), yarg_model_cond)
logprior(res), loglikelihood(res), parameters(res)
```

Now the only latent variable is `x`, and the `y` array is observed.

Conditioning on a model argument replaces the argument.
Inside the model, `y` is `[1.0, 2.0, 3.0]` from the first line, and the values in `my_y` are never used.

We can do the same with `ydef`.
Here `y = zeros(3)` is only a template: as when sampling, the value of each `y[i]` arrives at its tilde-statement, so `y[i]` holds `0.0` until then.

```@example model
ydef_model_cond = ydef_model | (; y=[1.0, 2.0, 3.0])

res = simulate(Xoshiro(468), ydef_model_cond)
logprior(res), loglikelihood(res), parameters(res)
```

Only whole variables can be conditioned. Partial array or field conditioning is not supported.
To mix observed and latent values, use separate variables or condition whole variables in a submodel.

When a whole array is conditioned, every tilde-statement on it must index into it.
For example, conditioning `y` on a 2-element array in `ydef` makes `y[3] ~ Normal(x)` throw a `BoundsError` instead of creating a new latent variable.

!!! warning "Conditioned values are read-only"

    Wren does not copy conditioned values, so inside the model a conditioned argument is the same array that you passed to `|`.
    Mutating it inside the model mutates your data.

## Enforced observations

Sometimes you may want to enforce in a model that a quantity is observed.
However, you might not be able to provide this information at the time of model definition (for example, this might depend on the value of another variable).

Take for example:

```@example model
@wren function y_not_obs()
    x ~ Normal()
    y = x + 1
    y ~ Normal()
end
```

In this model, even though `y` is defined in the model, the subsequent statement `y ~ Normal()` will treat `y` as a latent variable, and will sample a new value for it, which will certainly not be equal to `x + 1`:

```@example model
simulate(y_not_obs())
```

To force `y` to be observed, Wren provides a macro `@observe`:

```@example model
@wren function y_obs()
    x ~ Normal()
    y = x + 1
    @observe y ~ Normal()
end

res = simulate(y_obs())
```

We can check that these values are consistent:

```@example model
loglikelihood(res) ≈ logdensityof(Normal(), parameters(res)[@vn(x)] + 1)
```

You can in principle use this to enforce that model arguments are observed:

```@example model
@wren function yarg_obs(y)
    x ~ Normal()
    for i in eachindex(y)
        @observe y[i] ~ Normal(x)
    end
end

rand(yarg_obs([1.0, 2.0, 3.0]))
```

However, note that once a statement has been marked with `@observe`, you cannot 'uncondition' it without modifying the model's stored AST itself.

## Independent and identically distributed draws

Array-valued data are often modelled as independent draws from a single distribution. Writing this as a loop over `@observe y[i] ~ dist` works, but evaluates one scalar density per element. Wren provides `@iid`, which lowers `@iid y ~ dist` to `y ~ Wren.iid(dist, size(y))`: one array-valued distribution of independent draws from `dist` with the same shape as `y`. Backends can specialise `Wren.iid` for their distribution types, so this also lets them evaluate the whole array at once.

```@example model
@wren function iid_obs(y)
    mu ~ Normal()
    @observe @iid y ~ Normal(mu)
end

y = [1.0, 2.0, 3.0]
res = simulate(iid_obs(y))
loglikelihood(res) ≈ sum(logdensityof.(Normal(parameters(res)[@vn(mu)]), y))
```

When specifying priors for independent groups `@iid mu ~ dist`, there is typically no way for `@iid` to derive `size(mu)` because `mu` is not yet defined. You can instead specify `size` via an argument:

```@example model
@wren function iid_prior(y, group)
    n_groups = maximum(group)
    @iid mu ~ Normal() size=n_groups
    @observe @iid y ~ Normal(mu)
end
```

!!! warning

    Should you need to combine `@observe` with `@iid` _and_ the `size` argument, it'll be necessary to pass `@iid` as the inner macro to ensure that `size` is picked up correctly.
    Otherwise, you can combine these macros in either order.

## [Batched tilde-statements](@id batched)

One of Wren's unique features is syntax-level batching of tilde-statements.
This is done using the `@batch` macro:

```@example model
@wren function batched(x, y)
    N = length(x)
    @batch for i in 1:N
        x[i] ~ Normal()
        y[i] ~ Normal(x[i])
    end
end
x, y = randn(1000), randn(1000)
model = batched(x, y) | (; y=y)
```

Batched for-loops have the same semantics as regular for-loops.
However, they are automatically lowered into a vectorised form, which is more or less analogous to

```julia
x[1:N] ~ MvNormal(Zeros(N), I)
y[1:N] ~ MvNormal(x[1:N], I)
```

When `N` is large this can lead to significant speedups.

**There are two important things to note about `@batch`.**

**Firstly,** this batching is only valid if each iteration of the loop is independent of the others.
By marking a loop with `@batch`, you are effectively asserting that this is the case.
Wren does *not* verify this for you!

As an example of an invalid use of `@batch`, consider the following:

```julia
@batch for i in 1:(N-1)
    x[i+1] ~ Normal(x[i])
end
```

This will be lowered into the equivalent of

```julia
x[(@. (1:(N-1)) + 1)] ~ MvNormal(x[1:(N-1)], I)
```

which has a different meaning: `x[i]` is now read before it is written to.

**Secondly,** the batching is contingent on Wren knowing what to do with the right-hand side of the expression.
Specifically, the right-hand side must be a function call; and additionally, there must be a custom method defined for `Wren.batched` which dispatches on the function on the right-hand side (in this case, `Normal`).
Unfortunately, this means that it is more or less impossible to batch user-defined functions.

In the case where Wren doesn't know how to batch the right-hand side, it will fall back to a regular product distribution, i.e.,

```julia
x[1:N] ~ product_distribution([Normal() for i in 1:N])
y[1:N] ~ product_distribution([Normal(x[i]) for i in 1:N])
```

which guarantees correctness, but might not be as efficient as the vectorised form.

!!! danger

    Batches are a highly experimental feature and might not work in all cases.
    In particular, please note the following limitations:

      - The loop must contain only tilde-statements (`@observe` tilde-statements are fine).
        This can in theory be relaxed but is not yet implemented.

      - The loop must be `for SYMBOL in EXPR`, where `SYMBOL` is a single variable.
        Thus, loops of the form `for (i, x) in enumerate(xs)` cannot be batched.
        This probably cannot ever be supported.

      - The only form of indexing allowed is `x[i]` (`i` being the loop variable; `x` can be any identifier).
        This can in theory be relaxed to admit more forms, such as `x[i, j]`, `x[f(i)]`, `x[y[i]]`, etc., but this is not yet implemented.

      - Batches will increase allocations and will almost certainly cause slowdowns on small loops.

## [Manual log-likelihood addition](@id addloglike)

!!! danger "We do NOT recommend this"

    We strongly discourage the use of `@addloglikelihood`. It should only be considered if the model would otherwise not work (if you run into such a case, please let us know in an issue).

Wren provides a macro, `@addloglikelihood [name=:name] expr`, which allows you to manually add `expr` to the model's log-likelihood.

```@example model
@wren function manual_likelihood()
    x ~ Normal()
    @addloglikelihood name=:xsq x^2
end

simulate(manual_likelihood())
```

The `name` argument is optional and Wren currently does not use it, but hopefully will be wired up to pointwise logdensity calculations in the future.
