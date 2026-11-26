# [Vectorisation](@id vectorisation)

Wren models can be converted into a vectorised form, i.e., one where the parameters are supplied as a vector rather than a `VarNamedTuple`.
This allows Wren models to be used with a wide range of other Julia packages which expect a vectorised interface (i.e., LogDensityProblems.jl).

We begin by setting up our now-familiar eight schools model.

```@example vecmodel
using Wren, Wren.Dists

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
```

## Introduction

Conceptually, vectorising a model requires two things:

 1. a mapping from the original model's parameters to ranges of indices in a vector; and
 2. transformation from the original model's parameter space to an unconstrained space (i.e., a space where all parameters can take any value in ℝ^n).

The latter is *technically* not mandatory, but in practice often desirable due to the fact that many inference algorithms work best in unconstrained space.

Vectorisation is done with the `VecModel` constructor, which takes a model and an `ADTypes.AbstractADType` object, constructs the above mapping, and stores it.
The latter allows you to calculate gradients of the log-density with respect to the vectorised parameters.

```@example vecmodel
using Enzyme

# DEFAULT_AD is AutoEnzyme(; mode=Reverse)).
vecmodel = VecModel(model, Wren.DEFAULT_AD)
```

## Automatic differentiation

Wren's 'default' AD backend is reverse-mode Enzyme: in general we recommend this for performance.

However, Wren does also use [DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/) under the hood, so you can pass any AD backend which is supported by DI.
These include (but are not limited to) the following.
All of the following are re-exported by Wren:

  - `AutoEnzyme(; mode=Forward)` or `AutoEnzyme(; mode=Reverse)` using [Enzyme.jl](https://enzyme.mit.edu/index.fcgi/julia/stable/);
  - `AutoMooncakeForward()` or `AutoMooncake()` using [Mooncake.jl](https://chalk-lab.github.io/Mooncake.jl/stable/).
  - `AutoForwardDiff()` using [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/);
  - `AutoReverseDiff()` using [ReverseDiff.jl](https://juliadiff.org/ReverseDiff.jl/stable/);

!!! note "You must explicitly load the AD backend"

    Because AD functionality is implemented in the form of Julia package extensions, you *must* explicitly load the AD backend you want to use before creating a `VecModel`.
    Wren does not automatically load any AD backend, because this would require Wren to incur a dependency on them.

You can also use `AutoReactant` to further accelerate the model evaluation using [Reactant.jl](https://enzymead.github.io/Reactant.jl/stable/).
Please see the [Working with Reactant](@ref reactant) page for more information.

If you are only interested in vectorisation and not AD, you can pass `nothing` as the second argument:

```@example vecmodel
noad_vecmodel = VecModel(model, nothing)
```

## Interacting with `VecModel`

This is mostly useful when you want to hand off a Wren model to a different library which expects a vector interface.
In particular, `VecModel`s obey the [LogDensityProblems.jl interface](https://github.com/tpapp/LogDensityProblems.jl), and thus can be used with a wide range of other Julia packages that expect this.

Probably the first thing to do is to draw a vectorised sample from a `VecModel` with [`vector_parameters`](@ref).
This draws from the prior of the underlying model:

```@example vecmodel
v = vector_parameters(vecmodel)
```

This is exactly analogous to calling `parameters(model)`.
For example, you can also pass as the second argument a `Random.AbstractRNG` to control the random number generation; or pass a `VarNamedTuple` to evaluate the model at a specific set of values.
The latter can be thought of as a way of _converting_ the `VarNamedTuple` into a vectorised form.

```@example vecmodel
ps = parameters(model)               # This is a VarNamedTuple in original model space.
v = vector_parameters(vecmodel, ps)  # Convert it to vectorised (and unconstrained) form.
```

Once you have a vectorised sample, you can call `simulate` and its derivatives to evaluate the model at that point:

```@example vecmodel
simulate(vecmodel, v)
```

!!! note "Log-prior is untransformed"

    When calling `simulate`, the log-prior returned does not include log-Jacobian terms, even if the `VecModel` was constructed with a transformation to unconstrained space.

## Log-densities

In addition to the above interface, `VecModel`s also have a *privileged* notion of *a* log-density.
That is, unlike the underlying model which tracks each probability component separately and does not emphasise one over the other, `VecModel`s have a single type of log-density which it knows how to calculate.

By default, this log-density is the *transformed log-joint*: that is, the log-prior plus the log-likelihood, minus the log-Jacobian of the transformation from the original parameter space to the unconstrained space.
This is precisely the log-density that is required by many inference algorithms which run in unconstrained space.

The reason for having this single notion of a log-density is to satisfy the LogDensityProblems.jl interface, which expects a mapping from ``\mathbb{R}^n`` (parameters) to ``\mathbb{R}`` (a single log-density value).
If you have a vector of parameters, you can calculate the log-density of these vectorised parameters, and their gradient:

```@example vecmodel
import LogDensityProblems as LDP

LDP.logdensity(vecmodel, v)               # Works without an adtype.
LDP.logdensity_and_gradient(vecmodel, v)  # Requires an adtype.
```

Wren in fact exports shorter functions for these, namely [`lp`](@ref) and [`lp_and_grad`](@ref), which are directly equivalent to the above.
Additionally, Wren provides an in-place gradient [`lp_and_grad!`](@ref) function, which may be useful for squeezing every last bit of performance out.
Note that LogDensityProblems.jl does not provide a corresponding interface function for `lp_and_grad!` (but if you have a good usecase for it, please consider [adding a comment on this issue](https://github.com/tpapp/LogDensityProblems.jl/issues/115)!).

!!! note "Accessing log-density components"

    If you need to access the log-density components separately when evaluating a `VecModel`, it is recommended to either use `simulate` (less efficient) or directly call [`WrenCore.evaluate`](@ref) with the appropriate accumulators.
    Documentation on this will be forthcoming but for now please see the source code.

By default, `VecModel`s are always run in [`UnconstrainedSpace()`](@ref): that is, all parameters are transformed into a form where they can take any value in ℝ^n.
As mentioned above, the log-density calculation will then include the log-Jacobian term corresponding to this transformation.
If this is not desired, you can specify a different parameter space, or a different log-density calculation function as a keyword argument to the [`VecModel`](@ref) constructor: please see its docstring for more info.

!!! note "Higher-order derivatives"

    If you have a use for higher-order derivatives like Hessians, please open an issue!

## Mutation semantics

Note that, for performance reasons, Wren retains the right to mutate the input vector passed into `simulate`, `logdensity`, and other functions.
This is subject to the same considerations as outlined in [Working with `Model`s](@ref copying-arguments).

It's probably worth illustrating this with a simple example.
The easiest way to trigger this behaviour is if your model explicitly mutates arrays (e.g. writing `x[1] += 1` if `x` is a parameter).

```@example vecmodel
@wren function with_mutation()
    x ~ MvNormalScale(zeros(2), I)
    x[1] += 1.0
    @info x
end

vm = VecModel(with_mutation(), nothing)
```

```@example vecmodel
for i in 1:2
    vector_parameters(vm)
end
```

Each time we run the model, we get a new value of `x`, and mutate it inside the model.
So far, this looks reasonable.
However, the following can be quite surprising:

```@example vecmodel
p = [2.0, 3.0]
LDP.logdensity(vm, p)
p
```

The vector `p` has been mutated in-place by the `x[1] += 1.0` statement.

!!! note "The results are still correct"

    There isn't a *correctness* issue here at all: the log-density is always calculated with respect to the values that are passed in.
    The slightly surprising thing is that the values that are passed in are themselves mutated.

The reason for this is because `lp` is a model _evaluation_, and as discussed in [Working with `Model`s](@ref copying-arguments), Wren makes no promises to copy anything during model evaluation.

However, it is still always guaranteed that at model construction time the arguments to the model will be copied.
This holds true for `VecModel`s as well (if only because to make a `VecModel` you must first make a `Model`).
The parameter vector `p` above does *not* fall into this category: it does not exist at model construction time.

## Restrictions

Vectorisation places certain restrictions on the structure of the model.
In particular, it is not valid for a model to have a varying dimension.
This can be caused either by parameters which have varying sizes:

```julia
x ~ Poisson(3.0)
y ~ MvNormalScale(zeros(x), I)  # dimension of `y` is not constant
```

or models which simply have a varying number of parameters:

```julia
if x > 4
    z ~ Normal()  # existence of `z` is not constant
end
```

Names can differ between mutually exclusive branches while sharing a vector slot:

```@example vecmodel
@wren function branch_names()
    x ~ Normal()
    if x > 0
        y ~ Normal()
    else
        z ~ Normal()
    end
end

initial = FromValues(VarNamedTuple(; x=1.0, y=2.0))
vm = VecModel(branch_names(), Wren.DEFAULT_AD; latent_source=initial)
lp_and_grad(vm, [-1.0, 2.0])  # The second coordinate now supplies z.
```

Wren infers a shared slot when a straight-line top-level `if`/`else` has one differing latent name in each arm.
Those names must not be sampled elsewhere. Names common to both arms keep their slots even when their order changes.
For control flow that Wren cannot resolve, declare the groups explicitly:

```julia
VecModel(model, adtype; aliases=((@vn(y), @vn(z)),))
```

Each group must sample exactly one member per evaluation, with the same vector width and representation.
Aliases support plain top-level distribution latents in `OriginalSpace()` or `UnconstrainedSpace()`.
An explicit tuple replaces inference; `aliases=()` disables it.
Names outside these groups must remain fixed, and the total vector dimension must remain constant.
