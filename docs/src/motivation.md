# [Motivation](@id motivation)

Wren's design is mostly inspired by two precursors, [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl) and [Tilde.jl](https://github.com/cscherrer/Tilde.jl).
Much of the model evaluation machinery has been borrowed from DynamicPPL, which I worked on for two years.
However, there are a number of motivations which led me to write Wren:

## Cleanup

DynamicPPL has a fair amount of legacy code and concepts.
`VarInfo` is a fine example of this.
In fact, I put in a lot of work to refactor Turing to not need it, and in principle it can be removed quite easily now (see [DynamicPPL#1376](https://github.com/TuringLang/DynamicPPL.jl/issues/1376)).
In some ways Wren is an opportunity to have a fresh start and not carry over any unnecessary baggage.
Another example is DynamicPPL's idea of 'raw values', which are actually just parameter values, i.e., `params`!
The name came about because `VarInfo` used to always unconditionally store variables in vectorised form, and thus an adjective was required to distinguish the unvectorised values from the vectorised ones.
(In fact it used to be called 'values as in model', which was even worse.)
There are numerous other instances which I could go on about, but I won't bore you.

## Acceleration (1)

The fastest AD backend for DynamicPPL [across a number of benchmarks](https://juliabayes.org/ADTests) is reverse-mode Enzyme.
However, even this is not optimal: due to the way DynamicPPL models are constructed, running it (*technically*, any model with heap-allocated arguments such as a vector) requires enabling Enzyme's runtime activity analysis.

Wren doesn't require runtime activity, which can lead to nontrivial speedups (around 30–40%) with Enzyme.
This should already lead to concrete improvements for many standard use cases.

Although I *recommend* using Enzyme (please do open issues if it fails!), Wren does in fact use DifferentiationInterface under the hood, so you will always be able to swap in your favourite alternate AD backend if you want to.

## Acceleration (2)

Because of the performance improvements over the past year or so, DynamicPPL / Turing essentially has near-optimal performance in that it introduces almost no overhead relative to plain Julia code.
See, for example, these PRs: [DynamicPPL#1125](https://github.com/TuringLang/DynamicPPL.jl/pull/1125), [DynamicPPL#1132](https://github.com/TuringLang/DynamicPPL.jl/pull/1132), [DynamicPPL#1183](https://github.com/TuringLang/DynamicPPL.jl/pull/1183), [DynamicPPL#1211](https://github.com/TuringLang/DynamicPPL.jl/pull/1211).

However, plain Julia code itself *can still be accelerated further*, notably using [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), which traces Julia code into MLIR and then XLA (conceptually, this is what JAX does for Python).
This is especially important for large models and datasets, which are *exactly* the interesting use cases.
It's, frankly, not very useful to make a PPL that can run small models faster than Stan, because Stan *is* already good enough for small models.

Wren makes no compromise on performance for small models but has an explicit aim to be compatible with Reactant.jl.
This should, in principle, also allow Wren models to be executed on GPUs and TPUs, although I don't have the resources to test that myself at present.
It should also in principle allow vectorised / batched model evaluation, although this has not yet been tested.

## Compilation

Like [Tilde.jl](https://github.com/cscherrer/Tilde.jl) (and its precursor [Soss.jl](https://github.com/cscherrer/Soss.jl)), Wren performs compilation in two stages.
Firstly, Wren's `@wren` macro parses the model body into a probabilistic representation, and stores it inside the `Model` object.
For example, `@wren f() = x ~ Normal()` is represented as something like:

```julia
Model(:f, [Sample(:x, :(Normal()))])
```

Only afterwards is the evaluation function for the model generated (by traversing this AST and emitting the corresponding code).
This functionality depends on [RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl).

!!! note

    Soss and Tilde use [GeneralizedGenerated.jl](https://github.com/JuliaStaging/GeneralizedGenerated.jl) instead of RGF.
    GG is supposedly more powerful than RGF, but I haven't yet come across a use case where GG was necessary.
    Furthermore, RGF is actually maintained (for example it is used in ModelingToolkit.jl), and so it seems like a better choice for a dependency.

Since the AST is in principle capable of capturing all Julia code, this approach is no less flexible than DynamicPPL, but additionally allows for several things (which are not yet implemented, but are planned):

  - Higher-level probabilistic primitives, such as a `batch` construct which is used to mark conditionally independent loop iterations. See [issue #1](https://github.com/JuliaBayes/Wren.jl/issues/1) for a motivation of this.

  - Programmatic construction of models. A user can, instead of writing Julia code, manually build up a Wren AST and generate a model with it.

  - Programmatic modification of models. Once constructed, a user can reach into the AST and modify it. The evaluation function will then be automatically regenerated. (For example, if you don't like the way an external package defines their submodel, you can modify it!)

  - Static analysis of the model, for example, to extract a graphical representation. (See, e.g., [GraphPPL.jl](https://github.com/ReactiveBayes/GraphPPL.jl) for an existing implementation of this.) This might open up the possibility of more advanced inference algorithms which make use of this graphical representation, such as message passing.

This is a key difference from DynamicPPL, whose `@model` macro _directly_ emits the code of the evaluation function without storing it.
Consequently, DynamicPPL models are completely opaque once they have been constructed, and the only way to extract information from them is to actually run the model.

## Submodels

Wren's submodel interface is designed to be flexible.
In particular, submodel evaluation is executed in a context-free manner, meaning that there is no need for a submodel to know the way in which it is being called.
This is in contrast to DynamicPPL where evaluation context is passed into the submodel.

The outcome of this is that it is (or will be) possible to have a well-defined (if sometimes tedious) interface for submodels in Wren, which allows you to embed other probabilistic models or functions inside a Wren model.

On top of that, Wren more clearly separates the return value of a submodel from its latent variables.
DynamicPPL conflates these two (see [Turing#2485](https://github.com/TuringLang/Turing.jl/issues/2485), and also [Turing's docs page on submodels](https://turinglang.org/docs/usage/submodels/#submodels-versus-distributions), which I wrote).
