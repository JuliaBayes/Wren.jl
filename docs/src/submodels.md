# [Submodels](@id submodels)

Wren allows models to be nested within other models.
This is useful for building complex models from simpler components, and for reusing models in different contexts.

## Basic example

Suppose that we have the following inner model, which we'll embed inside an outer model:

```@example submodel
using Wren, Distributions

@model function inner()
    x ~ Normal()
end

rand(inner())
```

When we write `x ~ Normal()` in the model above, what happens is that the result of `rand(Normal())` (i.e., a float) is stored in the parameter `x`.

Now in the next model, we'll write `a ~ inner()`, with a model on the right-hand side of `~` instead of a distribution:

```@example submodel
@model function outer()
    a ~ inner()
end
```

In an exactly analogous fashion to before, the result of `rand(inner())` (i.e., a VarNamedTuple) is stored in the parameter `a`.
We could equivalently say that all the variables inside `inner` are *prefixed* with `a`, but thinking of it in terms of `rand()` is more precise.

```@example submodel
vnt = rand(outer())
```

We can access the inner variable by indexing into this:

```@example submodel
vnt[@vn(a.x)]
```

## Unprefixed submodels

TODO, see https://github.com/JuliaBayes/Wren.jl/issues/4.

## Accessing latent values

We claimed above that `a ~ inner()` should store the result of `rand(inner())` in the variable `a`.
In the examples above, we demonstrated this by evaluating the entire parent model end-to-end.
However, one might also want to access the latent variables of a submodel *inside* the parent model, during its execution.

You can do this directly by indexing into `a` with the relevant `VarName`:

```@example submodel
@model function outer()
    a ~ inner()
    ax = a[@vn(x)]
    # Make `y`'s support dependent on the value of `a.x`:
    y ~ Uniform(ax, ax + 0.01)
end

rand(outer())
```

!!! note "Performance"

    In general, this means that the submodel *must* be evaluated in a way where it also collects its parameter values (a 'trace').
    This is not always the case when evaluating models: for example if you only care about log-probabilities, you can get away without collecting a trace.
    The evaluation of submodels therefore comes with a performance cost: this is usually very small but can depend on the contents of the submodel.

    This performance cost can in principle be elided if Wren can prove that the submodel's latents are never used again in the parent model.
    Luckily, Wren models store their own AST so this is something that should be very possible to do!

## Accessing return values

With a distribution on the right-hand side of `~`, `rand(dist)` is indeed the _only_ output that we are interested in.
However, with a submodel, we may also be interested in the return value of the submodel, which is not part of the latent variables.

In order to allow users to access the return value, Wren stipulates that the `prefix ~ submodel` expression _evaluates_ to the return value of the submodel.
Thus, you can write `retval = (prefix ~ submodel)`:

!!! note

    The parentheses are not strictly required because `~` has higher precedence than `=`, but it's clearer to include them.

```@example submodel
@model function inner_with_retval()
    x ~ Normal()
    return x^2
end

@model function outer_with_retval()
    xsq = (a ~ inner_with_retval())
    @observe xsq ~ Normal()
end

accs = rand_with_logp(outer_with_retval())
```

The value of `xsq` should be the latent `a.x` squared, so we can check that the likelihood matches:

```@example submodel
xsq = params(accs)[@vn(a.x)]^2
logdensityof(Normal(), xsq) ≈ loglikelihood(accs)
```

(You can also use this syntax for distributions, but it's much less interesting as the `retval` will also be the same as `rand(dist)`.)
