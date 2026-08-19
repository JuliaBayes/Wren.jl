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

When we write `x ~ Normal()`, what happens is that the result of `rand(Normal())` (i.e., a float) is stored in the parameter `x`.

Likewise, when we write `a ~ inner()`, the result of `rand(inner())` (a VarNamedTuple) is stored in the parameter `a`.

We could equivalently say that all the variables inside `inner` are *prefixed* with `a`, but the description in terms of `rand()` is clearer (in my opinion).

```@example submodel
@model function outer()
    a ~ inner()
end

vnt = rand(outer())
```

We can access the inner model by indexing into this:

```@example submodel
vnt[@vn(a.x)]
```

## Unprefixed submodels

TODO, see https://github.com/JuliaBayes/Wren.jl/issues/4.

## Accessing latent values

TODO, see https://github.com/JuliaBayes/Wren.jl/issues/3.

## Accessing return values

With a distribution on the right-hand side of `~`, `rand(dist)` is indeed the _only_ output that we are interested in.
However, with a submodel, we may also be interested in the return value of the submodel, which is not part of the latent variables.

In order to allow users to access the return value, Wren stipulates that the `prefix ~ submodel` expression _evaluates_ to the return value of the submodel.
Thus, you can write `retval = (prefix ~ submodel)`:

!!! note

    The parentheses are not strictly required because `~` has lower precedence than `=`, but it's clearer to include them.

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
