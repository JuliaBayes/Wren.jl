# Staged computation

One of Wren's features, inspired by [Soss.jl](https://github.com/cscherrer/Soss.jl) and [Tilde.jl](https://github.com/cscherrer/Tilde.jl), is that it uses staged computation to generate the evaluation functions for models.
That sounds like a scary term, but it essentially means that the evaluation function is generated via a multi-stage pipeline:

```mermaid
flowchart TD
    A[Model definition] --> B[Probabilistic intermediate representation]
    B --> C[Evaluation function]
```

(Note that this intermediate representation, or IR, has nothing to do with the IR that Julia uses in its own compilation pipeline; this is entirely within Wren.)

The model definition is what you write:

```@example staged
using Wren, Wren.Dists

@wren function my_model(y)
    x ~ Normal()
    @observe y ~ Normal(x)
end

model = my_model(1.0)
```

The probabilistic IR looks something like the following, but a bit more complicated:

```julia
[Sample(:x, :(Normal())), Observe(:y, :(Normal(:x)))]
```

!!! details "The actual IR"

    Although it's not really useful for the average user, you can view this if you want with

    ```@example staged
    dump(model.stmts)
    ```

And the evaluation function is conceptually something like

```julia
function __mymodel_evaluate__(y, params, accs, ...)
    x = rand(Normal())
    lp_x = logdensityof(Normal(), x)
    accs = add_logprior(accs, lp_x)
    lp_y = logdensityof(Normal(x), y)
    accs = add_loglikelihood(accs, lp_y)
    return accs
end
```

!!! details "The actual evaluation function"

    Again, although it's not really useful for the average user, you can view this with

    ```@example staged
    using WrenCore
    
    WrenCore.view_evaluation_fn(model)
    ```

## Why do this?

This pipeline does seem complicated.
In particular, one might ask why we don't just generate the evaluation function directly from the model definition.
(Indeed, a macro would do this quite easily, and that's more or less how DynamicPPL works, for example.)

The main motivation is that the intermediate representation (IR):

  - gives Wren some information about the model which it can use to (more efficiently) generate the evaluation function;
  - allows Wren to do some optimisations, for example [elision of submodel parameter collection](@ref submodel-performance);
  - allows for the programmatic generation of Wren models, in that a user can construct an IR and then generate a Wren model from it, instead of having to do some macro gymnastics to create the original model definition.

!!! details "Example: generated quantities"

    TODO: Should we use generated quantities to illustrate staging or just leave this page as is?

This of course doesn't come for free.
Besides the substantial extra complexity of the codebase, it also means that we have to use [RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl) in the second step (IR to evaluation function), which brings with it some limitations.

!!! note

    Soss and Tilde use [GeneralizedGenerated.jl](https://github.com/JuliaStaging/GeneralizedGenerated.jl) instead of RGF.
    GG is supposedly more powerful than RGF, but I haven't yet come across a use case where GG was absolutely necessary.
    Furthermore, RGF is actually maintained (for example it is used in ModelingToolkit.jl), and so it seems like a better choice for a dependency.

For example, model bodies cannot close over local variables (only global variables).
This is surprisingly easy to trigger:

```@repl staged
for d in (Normal(), Exponential())
    # d is a local variable...

    @wren function oops_closed_over_local()
        # ... but our model closes over it
        x ~ d
    end

    model = oops_closed_over_local()
    parameters(model)
end
```

This can be worked around, for example by passing `d` into the model as an argument.
However, it's still of course a limitation, and the error message isn't super helpful either.

## IR and syntax design

Above we saw two of the 'special expressions' that Wren uses in its IR, namely `Sample` and `Observe`.
These have a very unimaginative one-to-one mapping to the special nodes in Wren:

| Syntax                  | IR node           |
|:----------------------- |:----------------- |
| `a ~ b`                 | Sample            |
| `@observe a ~ b`        | Observe           |
| `@iid size=N ...`       | IID               |
| `@batch for ...`        | Batch             |
| `@addloglikelihood ...` | AddLogLikelihood  |
| `@gq ...`               | GeneratedQuantity |
| `return ...`            | ReturnStatement   |

As you can see, most of these are denoted by macros.
As it turns out, none of these macros actually have a meaning outside of Wren model bodies: they don't actually expand into anything.
They simply serve as syntax markers for the IR generation stage.

Why macros, though?
To begin with, we are fundamentally limited by Julia syntax: we can't invent our own syntax, because it has to be parsed by a normal Julia macro (`@wren`).
So we can't, for example, write `[observe] a ~ b` as that simply won't parse.

Then, we don't want to hijack any preexisting Julia syntax.
This means that, for example, we don't want to parse `a => b` into something, because that usually means `Pair(a, b)` in Julia.
Once we eliminate those, we are actually left with very few operators that aren't weird Unicode stuff that is impossible to type.

In principle, we could use any kind of function call to denote these.
For example, we could say that `sample(a, b)` is what is now `a ~ b`, and `iid(sample(a, b); size=N)` is what is now `@iid a ~ b size=N`.
However, this is a bit too understated: it looks just like a normal function call, and it's not obvious to the user that these have special semantics.
On top of that, it actually stops the user from using any function named `sample` or `iid` in their model.

Macros are the best compromise between all of these.
We aren't restricted to a fixed set of possibilities (as is the case with operators), but also to most Julia programmers it should be intuitive that macros have special effects!
It does preclude the use of macros with the same name, but that is quite unlikely to be a problem as there are much fewer macros than functions in Julia.

## Programmatic generation

The above claim that the IR *allows* for programmatic generation is very carefully worded: there is no meaningful API for this yet.

However, you can in principle do something like this:

```@example staged
using WrenCore.Expressions: Sample, Observe, AbstractWrenExpr
using WrenCore: RuntimeGeneratedFunctions

stmts = AbstractWrenExpr[
    Sample(:sample_x, :x, Normal(), :x, true),
    Observe(:observe_y, :y, :(Normal(x))),
]

RuntimeGeneratedFunctions.init(@__MODULE__)
manual_model = Wren.Model(:my_model, (; y=1.0), stmts, @__MODULE__)
```

Obviously, this leaks internal details of Wren, and is not exactly a user-friendly API.
However, it does show that it should be possible to construct the `stmts` array programmatically, generate a Wren model from it, and then use that model:

```@example staged
parameters(manual_model)
```
