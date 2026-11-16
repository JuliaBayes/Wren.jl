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
model = eight_schools(y, sigma, J)
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

## When does Wren copy arguments?

...

Finally, we note here that when encoutering a tilde, the value of the left-hand side is overwritten with the value that is drawn from the distribution on the right-hand side.
This is a no-op for observations (because the left-hand side will _already_ have the value that it is observed to be).
However, for parameters, this means that execution of a Wren model can

## [Vectorisation](@id vectorisation)

Wren models can be converted into a vectorised form, i.e., one where the parameters are supplied as a vector rather than a `VarNamedTuple`.
This is done with the `VecModel` constructor, which takes a model and an `AbstractADType` object:

```julia
using Enzyme

vecmodel = VecModel(model, AutoEnzyme(; mode=Reverse))
```

If you are only interested in vectorisation and not AD, you can pass `nothing` as the second argument:

```julia
noad_vecmodel = VecModel(model, nothing)
```

This is mostly useful when you want to hand off a Wren model to a different library which expects a vector interface.
In particular, `VecModel`s obey the [LogDensityProblems.jl interface](https://github.com/tpapp/LogDensityProblems.jl), and thus can be used with a wide range of other Julia packages that expect this.

You can draw a vectorised sample from a `VecModel` with [`vector_parameters`](@ref):

```julia
v = vector_parameters(vecmodel)
```

or you can convert existing parameters into a vectorised form with [`vector_parameters`](@ref):

```julia
ps = parameters(model)               # This is a VarNamedTuple in original model space.
v = vector_parameters(vecmodel, ps)  # Convert it to vectorised (and unconstrained) form.
```

Then you can calculate the log-density of these vectorised parameters, and their gradient:

```julia
import LogDensityProblems as LDP

LDP.logdensity(vecmodel, v)               # Works without an adtype.
LDP.logdensity_and_gradient(vecmodel, v)  # Requires an adtype.
```

!!! note

    If you have a use for higher-order derivatives like Hessians, please open an issue!

Wren in fact exports shorter functions for these, namely [`lp`](@ref) and [`lp_and_grad`](@ref), which are directly equivalent to the above.
Additionally, Wren provides an in-place gradient [`lp_and_grad!`](@ref) function, which may be useful for squeezing every last bit of performance out.
Note that LogDensityProblems.jl does not provide a corresponding interface function for `lp_and_grad!` (but if you have a good usecase for it, please consider [adding a comment on this issue](https://github.com/tpapp/LogDensityProblems.jl/issues/115)!).

By default, `VecModel`s are always run in [`UnconstrainedSpace()`](@ref): that is, all parameters are transformed into a form where they can take any value in ℝ^n.
The log-density calculation will then include the log-Jacobian term corresponding to this transformation.
If this is not desired, you can specify a different parameter space, or a different log-density calculation function as a keyword argument to the [`VecModel`](@ref) constructor: please see its docstring for more info.

!!! warning "Vectorisation restrictions"

    Vectorisation places certain restrictions on the structure of the model.
    In particular, it is not valid for a model to have a varying dimension.
    This can be caused either by parameters which have varying sizes, or models which simply have a varying number of parameters:

    ```julia
    x ~ Poisson(3.0)
    y ~ MvNormal(zeros(x), I)  # dimension of `y` is not constant
    
    if x > 4
        z ~ Normal()  # existence of `z` is not constant
    end
    ```

    The set of variable names must also be constant across all model evaluations.
    Thus, even though the following model has a constant number of parameters, it is still not valid for vectorisation:

    ```julia
    x ~ Normal()
    if x > 0
        y ~ Normal()
    else
        z ~ Normal()
    end
    ```
