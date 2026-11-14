# [Defining models](@id defining)

On this page, we show how to define a probabilistic model with basic Wren syntax.

This page assumes that you know what a probabilistic model and its various components (e.g. parameters, observations) are.
If you are not familiar with these concepts, you can read our [introduction to probabilistic programming](@ref beginner) first.

## Model bodies are functions

Wren allows you to define a probabilistic model as a function, but prefixed with the `@wren` macro:

```@example defining
using Wren, Wren.Dists

@wren function my_model(m)
    x ~ Normal(m)
    return nothing
end
```

The function body can be thought of as the probabilistic model itself: it provides a complete specification of how to perform forward simulation.
Here, our forward simulation says: draw a random quantity `x` from a normal distribution centred at `m`, and then return `nothing`.

The body of a Wren model is plain Julia code, with some extra constructs that allow you to define probabilistic aspects of the model (which we'll explain later on this page).
A corollary of this is that you can use almost any library inside a Wren model.

!!! note "Function arguments are not special"

    If you have used Turing.jl before, you may know that arguments have a special meaning there in that they are observations (unless passed as `missing`).

    This is not the case in Wren.
    In Wren, **arguments have no special meaning**: they are just quantities that can be used inside the model body.

## Instantiating a model

To actually run this model, you will first have to create a [`Model`](@ref) struct, which can be thought of as Wren's own representation of a probabilistic model.
To create one of these `Model`s, you need to call the function, passing in any required arguments.

Calling a normal Julia function will run its body and give you its return value (here `nothing`).
However, with the `@wren` macro, this is different: the function is rewritten such that calling it instead returns a `Model` object.

```@example defining
model = my_model(0.0)
```

As we can see, this `Model` is a wrapper around its own arguments (here `m = 0.0`), plus the function body.
(We'll come to `conditioned` later on this page.)
At this point in time, the model itself has not been run at all.
We will be able to do so by passing the newly created `Model` to all the APIs of Wren.

On this page, we'll demonstrate only one function for interacting with a `Model`, which is [`simulate`](@ref).
When given a `Model` object, this performs the forward simulation of the probabilistic model, returning a [`ModelResult`](@ref) struct that contains all the outputs we might be interested in.

```@example defining
simulate(model)
```

This includes the parameters, associated probabilities, as well as the original return value of the function body.

## Random quantities

The most important custom syntax that Wren provides is the tilde operator (`~`).
Much like the corresponding mathematical notation, `x ~ dist` means that `x` is a random quantity drawn from the distribution `dist`.

### Left-hand sides

The left-hand side of a tilde can be either

  - a top-level variable, e.g., `x ~ Normal()`,
  - a field, e.g., `m.x ~ Normal()`,
  - an array element, e.g., `a[1] ~ Normal()`,
  - or chains thereof, e.g., `m.a[1].x ~ Normal()`.

These are all things which can be assigned to in Julia (sometimes called 'lvalues').
Other expressions, including literals, are not (yet?) permitted.

Note that if you have a field or element on the left-hand side, the parent object must already exist.
That is, you cannot write `m[1] ~ Normal()` or `m.x ~ Normal()` if `m` is not already defined.

### Right-hand sides

Here, 'distributions' refer to any object that implements `Base.rand(rng, dist)` and `DensityInterface.logdensityof(dist, x)`; the object doesn't have to subtype any abstract type.
This includes those defined in Distributions.jl, as well as [Wren's own distributions](@ref distributions).
Apart from distributions, it is also possible to put a Wren `Model` itself on the right-hand side of a tilde.
In such a case, the `Model` is referred to as a *submodel*.
Please see the [submodels](@ref submodels) page for more information.

## Parameters vs observations

`@observe`: unconditionally turn this into an observation. Currently no way to 'undo'

Conditioning

## Special nodes

Only need to make a quick note of it here, actual explanation can be deferred to another page.

- `@iid` and `@batch`
- `@gq`
- `@addloglikelihood`

## Some technical limitations

Briefly mention RGF's technical limitations:

- can't close over local variables in model body
- can't define closures in model body that close over outside variables (...? get Claude to check this)
