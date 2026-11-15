# [Syntax and semantics](@id syntax)

!!! danger "Due to be rewritten"

    penelopeysm: This page will be rewritten into something that is much more formal, like a grammar.


## Sample/observe

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

```julia
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

```julia
parameters(model);
model.args.y
```

and if we evaluate it again, the log-density associated with the `@observe` statement will change.

However, Wren guarantees that the value of `y_outside` will never be mutated.
This is because when Wren _constructs_ a model (i.e., when calling `mutable(y_outside)`), it will copy the value of `y_outside`.

```julia
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

```julia
yarg_model_cond = yarg_model | (; y=[1.0, 2.0, 3.0])

res = simulate(yarg_model_cond, Xoshiro(468))
logprior(res), loglikelihood(res), parameters(res)
```

Now the only latent variable is `x`, and the `y` array is observed.

Conditioning on a model argument replaces the argument.
Inside the model, `y` is `[1.0, 2.0, 3.0]` from the first line, and the values in `my_y` are never used.

We can do the same with `ydef`.
Here `y = zeros(3)` is only a template: as when sampling, the value of each `y[i]` arrives at its tilde-statement, so `y[i]` holds `0.0` until then.

```julia
ydef_model_cond = ydef_model | (; y=[1.0, 2.0, 3.0])

res = simulate(ydef_model_cond, Xoshiro(468))
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

```julia
@wren function y_not_obs()
    x ~ Normal()
    y = x + 1
    y ~ Normal()
end
```

In this model, even though `y` is defined in the model, the subsequent statement `y ~ Normal()` will treat `y` as a latent variable, and will sample a new value for it, which will certainly not be equal to `x + 1`:

```julia
simulate(y_not_obs())
```

To force `y` to be observed, Wren provides a macro `@observe`:

```julia
@wren function y_obs()
    x ~ Normal()
    y = x + 1
    @observe y ~ Normal()
end

res = simulate(y_obs())
```

We can check that these values are consistent:

```julia
loglikelihood(res) ≈ logdensityof(Normal(), parameters(res)[@vn(x)] + 1)
```

You can in principle use this to enforce that model arguments are observed:

```julia
@wren function yarg_obs(y)
    x ~ Normal()
    for i in eachindex(y)
        @observe y[i] ~ Normal(x)
    end
end

parameters(yarg_obs([1.0, 2.0, 3.0]))
```

However, note that once a statement has been marked with `@observe`, you cannot 'uncondition' it without modifying the model's stored AST itself.

## Independent and identically distributed draws

Array-valued data are often modelled as independent draws from a single distribution. Writing this as a loop over `@observe y[i] ~ dist` works, but evaluates one scalar density per element. Wren provides `@iid`, which lowers `@iid y ~ dist` to `y ~ Wren.iid(dist, size(y))`: one array-valued distribution of independent draws from `dist` with the same shape as `y`. Backends can specialise `Wren.iid` for their distribution types, so this also lets them evaluate the whole array at once.

```julia
@wren function iid_obs(y)
    mu ~ Normal()
    @observe @iid y ~ Normal(mu)
end

y = [1.0, 2.0, 3.0]
res = simulate(iid_obs(y))
loglikelihood(res) ≈ sum(logdensityof.(Normal(parameters(res)[@vn(mu)]), y))
```

When specifying priors for independent groups `@iid mu ~ dist`, there is typically no way for `@iid` to derive `size(mu)` because `mu` is not yet defined. You can instead specify `size` via an argument:

```julia
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

```julia
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

## [Generated quantities](@id generated-quantities)

After sampling from a model, we often want to transform these samples into other values we're interested in.
Following the nomenclature established by Stan, we call these "generated quantities" (GQs).

For example, consider logistic regression where parameters are often on the log-odds scale.
Reporting log-odds is nasty, so we instead transform them to odds ratios:

```julia
@wren function logistic_regression(y)
    beta ~ Normal()
    for i in eachindex(y)
        @observe y[i] ~ BernoulliLogit(beta)
    end
    @gq odds_ratio = exp(beta)
end

y = [0, 1, 0, 0, 0, 1]
model = logistic_regression(y)
simulate(model)
```

You can write any expression on the right-hand of `@gq`, including `begin` or `let` blocks.
However, naming generated quantities is more restrictive: a name must be a single symbol, so names like `@gq x[1]` or `@gq x.a` are not allowed.

!!! note "Generated quantities are not copied"

    After initialising a generated quantity `@gq my_array`, mutating `my_array` will change its values. To create a snapshot of `my_array` at the time it is initialised, you need to copy explicity `@gq snapshot = deepcopy(my_array)`.

By default, all GQs are calculated and returned.
You can, however, omit the statements marked with `@gq` - for example because they're expensive and not needed at this time - by passing `gqs=NoGQs()` to `simulate`, `prior` or `nuts`. If you want to keep specific GQs, use `SomeGQs(:x)` instead.

!!! note "No side effects... again"

    When the definition of a generated quantity `x` involves side effects, such as calling `push!` or drawing a random number, and `x` is omitted via `gqs`, the side effects in its definition will not happen!

## [Manual log-likelihood addition](@id addloglike)

!!! danger "We do NOT recommend this"

    We strongly discourage the use of `@addloglikelihood`. It should only be considered if the model would otherwise not work (if you run into such a case, please let us know in an issue).

Wren provides a macro, `@addloglikelihood expr [name=:name]`, which allows you to manually add `expr` to the model's log-likelihood.

```julia
@wren function manual_likelihood()
    x ~ Normal()
    @addloglikelihood x^2 name=:xsq
end

simulate(manual_likelihood())
```

The `name` argument is optional and in fact Wren currently does not use it.
However, hopefully this will in future be wired up to pointwise logdensity calculations in the future.
