# [Working with Reactant](@id reactant)

[Reactant.jl](https://enzymead.github.io/Reactant.jl/stable/) is a Julia package which compiles Julia code into lower-level representations (MLIR and XLA) that can be run on accelerators like GPUs, often with very significant speedups.

One of the core design goals of Wren is to be maximally compatible with Reactant.
We acknowledge that this can be difficult: Reactant itself is a very ambitious project, and still has some rough edges.
Furthermore, Wren models can contain almost arbitrary code, and in general it is not Wren's responsibility to ensure that all of this code is compatible with Reactant.

However, at the same time, we strongly believe that this will help to push the Julia ecosystem forward and make Reactant work better for everybody.
To this end, we ask you to please [report any issues](https://github.com/JuliaBayes/Wren.jl/issues) you encounter when using Wren with Reactant.
We can help to minimise issues, file them upstream, and in some cases even fix them.

## Using Reactant with Wren

Currently, Reactant is only compatible with vectorised Wren models, i.e., the `VecModel` struct.
However, because this struct obeys the LogDensityProblems.jl interface, this does in fact allow for compatibility with a wide range of packages that expect this interface.
In particular, you can use Reactant with Wren's [`nuts`](@ref) sampler (which uses DynamicHMC under the hood).

To enable Reactant, you have to pass `AutoReactant{<:AutoEnzyme}` as the `adtype` keyword argument.

TODO example

Note that the first compilation will take a while as Reactant has to trace the model execution and generate code.

## Do I need to call `to_rarray`?

TODO write

## Host / device transfer

TODO discuss

## Reactant's `@trace` macro

Users of Reactant may be familiar with `@trace`d for loops or conditionals.

Unfortunately, one current restriction of Wren is that you cannot use `@trace` inside a Wren model.

The reason for this is incredibly technical.
As described in [the compilation model](@ref compilation), Wren uses RuntimeGeneratedFunctions.jl to create the model evaluation function.
RGF, or specifically its `@RuntimeGeneratedFunction` macro, must be able to 'see' the closure in order to convert it to a Julia opaque closure.
However, Wren does not expand macros in the body before passing it to `@RuntimeGeneratedFunctions`.

It's very possible that this can be fixed: apparently, [Claude managed to do it](https://github.com/JuliaBayes/Wren.jl/issues/293), so a human just has to get round to checking it.

## Troubleshooting

Extremely long compilation times (possibly use trace)

Allowscalar
