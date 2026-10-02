# [Submodels](@id submodels)

Wren allows models to be nested within other models.
This is useful for building complex models from simpler components, and for reusing models in different contexts.

## Basic example

Suppose that we have the following inner model, which we'll embed inside an outer model:

```@example submodel
using Wren, Wren.Dists

@wren function inner()
    x ~ Normal()
end

rand(inner())
```

When we write `x ~ Normal()` in the model above, what happens is that the result of `rand(Normal())` (i.e., a float) is stored in the parameter `x`.

Now in the next model, we'll write `a ~ inner()`, with a model on the right-hand side of `~` instead of a distribution:

```@example submodel
@wren function outer_basic()
    a ~ inner()
end
```

In an exactly analogous fashion to before, the result of `rand(inner())` (i.e., a VarNamedTuple) is stored in the parameter `a`.
We could equivalently say that all the variables inside `inner` are *prefixed* with `a`, but thinking of it in terms of `rand()` is more precise.

```@example submodel
vnt = rand(outer_basic())
```

We can access the inner variable by indexing into this:

```@example submodel
vnt[@vn(a.x)]
```

## Unprefixed submodels

Currently it is not possible to use submodels without prefixing the inner variables.
This is intentionally done to avoid name collisions; if you have a strong use case for this please consider leaving a comment on [this issue](https://github.com/JuliaBayes/Wren.jl/issues/4)!

## Accessing latent values

We claimed above that `a ~ inner()` should store the result of `rand(inner())` in the variable `a`.
In the examples above, we demonstrated this by evaluating the entire parent model end-to-end.
However, one might also want to access the latent variables of a submodel *inside* the parent model, during its execution.

You can do this directly by indexing into `a` with the relevant `VarName`.
For example, here we access the latent variable `x` of the inner model and square it:

```@example submodel
@wren function outer_with_latent()
    a ~ inner()
    xsq = (a[@vn(x)])^2
    # Make `y`'s support dependent on the square of `a.x`:
    y ~ Uniform(xsq, xsq + 0.01)
end

rand(outer_with_latent())
```

## Accessing return values

With a distribution on the right-hand side of `~`, `rand(dist)` is indeed the _only_ output that we are interested in.
However, with a submodel, we may also be interested in the return value of the submodel, which is not part of the latent variables.

For example, in this submodel, the latents include `x`, but the return value is `x^2`:

```@example submodel
@wren function inner_with_retval()
    x ~ Normal()
    return x^2
end
```

In order to allow users to access the return value, Wren stipulates that the `prefix ~ submodel` expression _evaluates_ to the return value of the submodel.
Thus, you can write `retval = (prefix ~ submodel)`:

!!! note

    The parentheses are not strictly required because `~` has higher precedence than `=`, but it's clearer to include them.

```@example submodel
@wren function outer_with_retval()
    xsq = (a ~ inner_with_retval())
    y ~ Uniform(xsq, xsq + 0.01)
end
```

As in the previous example, `y` should be very close to `a.x`:

```@example submodel
rand(outer_with_retval())
```

(You can also use this syntax for distributions, but it's much less interesting as the `retval` will also be the same as `rand(dist)`.)

## Performance considerations

**Note that accessing the latent variables of a submodel comes with a performance cost, *especially* when using gradient-based samplers.
If at all possible, you should always use the return values.**

Looking at our examples again:

```julia
@wren function outer_with_latent()
    a ~ inner()
    xsq = (a[@vn(x)])^2
    # Do stuff with `xsq`...
end

@wren function outer_with_retval()
    xsq = (a ~ inner())
    # Do stuff with `xsq`...
end
```

As a reminder, `a` here is a `VarNamedTuple` containing the latent variables of `inner`.
In the second case, it's clear that the value of `a` is never needed.
**Wren will in such cases completely optimise away the computation of `a`.**

This often does not really save much time when calling e.g. `rand`, because `rand` forces the construction of a `VarNamedTuple` anyway.
However, this makes a big difference when calculating gradients, because when calculating gradients we're only interested in the log-density terms.
If you make the log-density term dependent on the `VarNamedTuple` of latents, then the AD backend has to differentiate through all of that!

```@example submodel
using Enzyme

# Note: `Wren.DEFAULT_AD` is Enzyme reverse-mode.
@bmodel outer_with_latent() DEFAULT_AD
```

Most of the benchmarks above (`rand(model)`, `logjoint(model, params)`, and `lp` i.e. a vectorised log-density) will not be very different when we use the `retval` version.
However, `lp_and_grad` i.e. the gradient of the vectorised log-density) will be *much* faster!

```@example submodel
@bmodel outer_with_retval() DEFAULT_AD
```
