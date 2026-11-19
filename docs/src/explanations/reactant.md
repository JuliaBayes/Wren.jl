# [Working with Reactant](@id reactant)

[Reactant.jl](https://enzymead.github.io/Reactant.jl/stable/) is a Julia package which compiles Julia code into lower-level representations (MLIR and XLA) that can be run on accelerators like GPUs, often with very significant speedups.

One of the core design goals of Wren is to be maximally compatible with Reactant.
We acknowledge that this can be difficult: Reactant itself is a very ambitious project, and still has some rough edges.
Furthermore, Wren models can contain almost arbitrary code, and in general it is not Wren's responsibility to ensure that all of this code is compatible with Reactant.

However, at the same time, we strongly believe that this will help to push the Julia ecosystem forward and make Reactant work better for everybody.
To this end, we ask you to please [report any issues](https://github.com/JuliaBayes/Wren.jl/issues) you encounter when using Wren with Reactant.
We can help to minimise issues, file them upstream, and in some cases even fix them.

## Distributions

Before we even begin to talk about models, we should note that Reactant *will* run into many problems with Distributions.jl.
There are many reasons for this, but the main one is that Reactant's tracing mechanism requires that functions accept subtypes of `Number`, and virtually all functions in Distributions require `Real` arguments.

The creation of [the `Wren.Dists` module](@ref wrendists) was almost completely motivated by this.
To make Wren models compatible with Reactant, we recommend that you use Wren's built-in distributions as far as possible.
If there isn't a distribution you need, do feel free to [open an issue or a pull request](https://github.com/JuliaBayes/Wren.jl)!

## Using Reactant with Wren

Currently, Reactant can only be used with vectorised Wren models, i.e., the `VecModel` struct.
Because this struct obeys the LogDensityProblems.jl interface, this does still allow for compatibility with a wide range of packages that expect this interface!
In particular, you can use Reactant with Wren's [`nuts`](@ref) sampler (which uses DynamicHMC under the hood).

For [`independent_mh`](@ref), [`rwm`](@ref), [`ram`](@ref), [`am`](@ref), and
[`dram`](@ref), pass `adtype=AutoReactant()` to compile density evaluation.
These samplers use a fixed continuous latent layout with Reactant and run transitions on the host.
Independent MH draws on the host and scores the likelihood on the device.
Its latent coordinates must share one floating-point type.

To use Reactant directly with `VecModel`, pass `AutoReactant()` as its second argument.
Let's define our eight-schools model yet again, but this time with Reactant enabled:

```@example reactant
using Wren, Wren.Dists, Reactant

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(0, 5)
    theta ~ MvNormalScale(fill(mu, J), tau * I)
    @batch for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

model = eight_schools(y, sigma, J)

vm = VecModel(model, AutoReactant(DEFAULT_AD))
```

Note that the first compilation can take a while as Reactant has to trace the model execution and generate code.

If you got this far, you should be able to call both `lp` and `lp_and_grad` on the `VecModel` object:

```@example reactant
p = vector_parameters(vm)  # Draw random sample
res1 = lp_and_grad(vm, p)  # Evaluate log density and gradient with Reactant
```

!!! note "LogDensityProblems.jl"

    The equivalent LogDensityProblems.jl interface functions also work the same way, so you can pass this Reactant-enabled `VecModel` object to any package that expects a LogDensityProblems.jl interface.

When using Reactant we recommend that you do check the results against a non-Reactant implementation, to make sure that the results are consistent.

```@example reactant
vm_base = VecModel(model, DEFAULT_AD) # For comparison.
res2 = lp_and_grad(vm_base, p)
(res1[1] ≈ res2[1], res1[2] ≈ res2[2])
```

## Host / device transfer

While the implementation of Reactant acceleration above *works*, it is in fact extremely naive.

Reactant works by compiling the model evaluation function into XLA code which is then executed on a *device* (such as a GPU; or if you are running this locally, maybe a different part of your CPU).
However, before it can execute this code, it has to transfer the input arguments from the *host* (your CPU) to the device.
Furthermore, once it has finished evaluating this model, it has to transfer the results back from the device to the host.

The implementation of `lp` and `lp_and_grad` above encompasses all of these steps (i.e., `lp` takes a host array, transfers it to the device, evaluates the model on the device, and then transfers the float result back to the host).
This transfer can be quite expensive, and can **easily** dominate the time taken to evaluate a small model.
Eight-schools firmly falls into the small-model category:

```@example reactant
@bmodel model AutoReactant(DEFAULT_AD) # With Reactant
```

```@example reactant
@bmodel model DEFAULT_AD # No Reactant
```

This means that in Wren's current state, Reactant is likely to be most useful for *large* models, such as those with large datasets / likelihood terms, where the fixed cost of transferring the data is small in comparison to the cost of evaluating the model.

It should be emphasised that this is not a fundamental limitation of Reactant.
Instead, it is a choice that is made in Wren in order to abstract away these details of host/device transfer from the user.
We would *very much like to change this in the future*, and allow `VecModel` to use device arrays as inputs and outputs.
The difficulty with this is that everything else that the `VecModel` slots into, such as a HMC or NUTS sampler, needs to be converted into a form that is also compatible with Reactant.
At present, to the best of our knowledge, we don't really have that available to us.

(Conceptually, what we would like to have is something like [the 'iterative NUTS' procedure developed for NumPyro](https://github.com/pyro-ppl/numpyro/wiki/Iterative-NUTS).
That is more or less the same thing as having a Reactant-compatible NUTS sampler, but for JAX instead of Reactant.)

## Do I need to call `to_rarray`?

When invoking Reactant outside of Wren you may be aware that you often need to call `Reactant.to_rarray` on input arguments so that they are 'tracked' by the Reactant compiler.
Wren handles this for you by _unconditionally_ calling `to_rarray` on all input arguments to the model, so you do not need to worry about this.

Semantically, this is not the right thing to do: if you have an argument which is constant (i.e., you never want to evaluate the model with a different value of that argument), then we should _not_ call `to_rarray` on it, as that causes extra work for Reactant.
Model arguments should in theory always fall into this category, since once the model has been created, the arguments can't be changed without creating a new model!

Unfortunately, at the present moment there are a handful of issues which cause Reactant to grind to a halt when performing matrix multiplications with untraced arrays.
(See the WrenCore source code for more details.)
It is our hope that these can be fixed soon, which will allow us to remove this unconditional `to_rarray` call and potentially improve performance.

## Reactant's `@trace` macro

Users of Reactant may be familiar with `@trace`d for loops or conditionals.

Unfortunately, one current restriction of Wren is that you cannot use `@trace` inside a Wren model.

The reason for this is incredibly technical.
As described in [the compilation model](@ref compilation), Wren uses RuntimeGeneratedFunctions.jl to create the model evaluation function.
RGF, or specifically its `@RuntimeGeneratedFunction` macro, must be able to 'see' the closure in order to convert it to a Julia opaque closure.
However, Wren does not expand macros in the body before passing it to `@RuntimeGeneratedFunctions`.

It's very possible that this can be fixed: apparently, [Claude managed to do it](https://github.com/JuliaBayes/Wren.jl/issues/293), so a human just has to get round to checking it.

## Troubleshooting

We hope to tabulate common issues when using Wren with Reactant here.
[You can help by telling us about issues you run into!](https://github.com/JuliaBayes/Wren.jl/issues)
