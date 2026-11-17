# [Working with `Model`s](@id using-models)

Once you have [defined a model](@ref defining-models), you can use it to draw samples, calculate log-densities, and more.

This page describes Wren's high-level interface for running models and extracting information from them.

To begin, let's define a model with several of the features that Wren supports:

```@example using-models
using Wren, Wren.Dists

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools_centred(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(0, 5)
    theta_trans ~ MvNormalScale(zeros(J), I)
    @gq theta = mu .+ tau .* theta_trans
    for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

model = eight_schools_centred(y, sigma, J)
```

## The `simulate` family

The core function that Wren provides for drawing samples from a model is [`simulate`](@ref).

Without any arguments, `simulate` will draw from the prior distribution of the model:

```@example using-models
res = simulate(model)
```

It returns a `ModelResult` object which has several fields.
To extract these, it is recommended to use the following accessor functions:

  - [`parameters`](@ref): returns a [`VarNamedTuple`](@extref) of the parameters drawn from the model;
  - [`generated_quantities`](@ref): returns a [`VarNamedTuple`](@extref) of the generated quantities that were saved;
  - [`return_value`](@ref): the return value (here a `NamedTuple` with `mu` and `tau`);
  - [`logprior`](@ref), [`loglikelihood`](@ref), and [`logjoint`](@ref): the log-density components.

For example:

```@example using-models
parameters(res)
```

This `VarNamedTuple` object can, roughly speaking, be thought of as a mapping from variable names (`VarName`s) to values.
For example, indexing into it works:

```@example using-models
parameters(res)[@vn(mu)]
```

If you want to perform forward simulation but you only want the parameters, you can directly call `parameters(model)`.
This will be more performant than `parameters(simulate(model))`, because it avoids calculating the log-densities.
(The other functions behave exactly the same way.)

```@example using-models
parameters(model)
```

The functions `parameters`, `generated_quantities`, `return_value`, `logprior`, `loglikelihood`, and `logjoint` can therefore be thought of in the following manner:

  - if you pass it a result of an evaluation, it will extract the corresponding field;
  - if you haven't yet evaluated the model, it will evaluate it and return the corresponding field.

## Specifying input sources

While the above is useful for quickly drawing random samples, often we want to control exactly how these values are drawn.

For example, we might want to draw from the prior using a specific RNG seed.
You can do this by passing a `Random.AbstractRNG` to `simulate`.

```@example using-models
using Random

simulate(model, Xoshiro(468))
```

!!! warning "API intentionally differs from other Julia libraries"

    Please note that the RNG is passed as the second argument, not the first.

All the other derived functions (`parameters`, `generated_quantities`, etc.) also behave the same way:

```@example using-models
parameters(model, Xoshiro(468))
```

Alternatively, you may want to evaluate the model at a specific set of values, rather than letting Wren draw them from the prior.
This can be done by passing a `VarNamedTuple` to `simulate`:

```@example using-models
pars = VarNamedTuple(mu=1.0, tau=2.0, theta_trans=randn(8))

simulate(model, pars)
```

!!! note "General latent sources"

    From the above, one can see that the RNG seed, or the parameter values, both play the same role: they are _sources_ of latent values for the model.
    Under the hood, Wren is actually converting these to an [`AbstractLatentSource`](@ref) object.
    Indeed, you can construct and pass one of these directly for even more control (for example, you might want to specify some parameters and sample others.)

    This is the reason why Wren's API specifies the seed as the second argument: it's because the second argument can be thought of as 'a latent source or something that can be converted to one'.

## VarName and VarNamedTuple

At Wren's core are two types: [`VarName`](@extref) and [`VarNamedTuple`](@extref).
(If you have used [FlexiChains.jl](https://juliabayes.org/FlexiChains.jl/stable/), you may have seen these before.)

`VarName` is a representation of a variable or part of it, for example: `x`, `x[1]`, or `x.a`.

```@example using-models
vn1 = @vn(x)     # a variable named `x`
vn2 = @vn(x[1])  # the first element of a variable named `x`
(vn1, vn2)
```

A `VarNamedTuple` is a mapping of `VarName`s to values, which has a Dict-like interface.
However, it is designed to be much more general and performant than a simple `Dict` or `NamedTuple`.

The most frequent operation you will encounter is indexing into it.
Returning to our model above:

```@example using-models
vnt = parameters(model)
```

we can access not only the top-level variables

```@example using-models
vnt[@vn(mu)], vnt[@vn(theta_trans)]
```

but also _parts_ of variables:

```@example using-models
vnt[@vn(theta_trans[1])]
```

Notice here that `@vn(...)` describes what we want to access (sort of a 'path' into the data that we're interested in).

!!! note "Constructing your own VarNamedTuple"

    Most often end-users who want to set specific values known in advance will want to construct a `VarNamedTuple` via the [`VarNames.@vnt`](@extref) macro.
    If you are writing an inference algorithm which has to work over arbitrary models, you should look at the lower-level interface, including [`VarNames.templated_setindex!!`](@extref).

    Usually, you will obtain `VarNamedTuple`s from Wren's or FlexiChain's functions, so you will not need to think too much about how to construct them, unless you are writing custom inference algorithms.
    For this reason, although we would like to have more docs on this in the future, for now we defer a detailed discussion of `VarNamedTuple` to the [VarNames.jl docs](https://juliabayes.org/VarNames.jl).

## [When does Wren copy arguments?](@id copying-arguments)

At this point, since we have been evaluating models a few times, it is an apposite time to discuss when Wren makes copies of the model arguments.

In general, for performance reasons, Wren tries to avoid copying its arguments as far as possible.
This means that if you pass a mutable object to Wren, it may be modified in-place during evaluation of a model.

Notice how, for example, `x[1]` is different on each iteration of the following loop:

```@example using-models
@wren function model_with_mutable_arg(x)
    @info ("before", x[1])
    x[1] ~ Normal(0, 1)
    @info ("after", x[1])
end

model = model_with_mutable_arg([0.0])

for i in 1:2
    simulate(model)
end
```

What happens here is that when a sampling statement is encountered, the resulting value is stored back into `x[1]`.
So the second time the model is evaluated, `x[1]` will have taken on the value that it was assigned in the first evaluation.

However, to try to minimise potential footguns, Wren will make a copy of arguments *at model construction time*.
The reason why this is permissible is because you will only construct the model once (as opposed to evaluation, which happens many times in performance-sensitive code).

In the following example, we do the same as above, but we give the argument a name so that we can check whether it has been modified:

```@example using-models
x_original = [0.0]

model = model_with_mutable_arg(x_original)

for i in 1:2
    simulate(model)
end
```

The behaviour so far is unsurprisingly exactly the same as above, in that `x[1]` is modified in-place during model evaluation.
However, if we check the original array passed into the model, we can see that it has not been modified, as Wren made a copy of it for the model:

```@example using-models
x_original
```
