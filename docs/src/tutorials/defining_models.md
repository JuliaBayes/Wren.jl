# [Defining models](@id defining)

On this page, we show how to define a probabilistic model with basic Wren syntax.

This page assumes that you know what a probabilistic model and its various components (e.g. parameters, observations) are.
If you are not familiar with these concepts, you can read our [introduction to probabilistic programming](@ref beginner) first.

## Model bodies are functions

Wren allows you to define a probabilistic model as a function body, but prefixed with the `@wren` macro:

```@example defining
using Wren, Wren.Dists

@wren function my_model(m)
    x ~ Normal(m)
    # other contents of the model ...
    return nothing
end
```

The function body can be thought of as a specification of how to perform the forward simulation of the probabilistic model.

To then instantiate an actual [`Model`](@ref) object, you need to call the function, passing in any required arguments.
This may be confusing, because calling a normal Julia function will give you its return value (here `nothing`).
However, in this case, the `@wren` macro overloads this to instead return the `Model` object.

```@example defining
model = my_model(0.0)
```

!!! note "Function arguments are not special"

    If you have used Turing.jl before, you may know that arguments have a special meaning there in that they are observations (unless passed as `missing`).

    This is not the case in Wren.
    In Wren, **arguments have no special meaning**: they are just quantities that can be used inside the model body.

As can be seen, this `Model` object contains its own arguments (here `m = 0.0`), plus the function body.
(We'll come to `conditioned` later on this page.)
It is this object that is then passed to all the APIs of Wren.

On this page, we'll demonstrate only one function for interacting with a `Model`, which is [`simulate`](@ref).
When given a `Model` object, this performs the forward simulation of the probabilistic model, returning a `ModelResult` object that contains all the outputs we might be interested in.

```@example defining
simulate(model)
```

## Ordinary Julia code

Blah

## Random quantities

Tildes

Briefly mention submodels, but defer to another page for details.

## Parameters vs observations

`@observe`: unconditionally turn this into an observation. Currently no way to 'undo'

Conditioning

## Special nodes

Only need to make a quick note of it here, actual explanation can be deferred to another page.

- `@iid` and `@batch`
- `@gq`
- `@addloglikelihood`
