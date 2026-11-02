# [Working with models](@id models)

Once you have defined a model you will likely want to do something with it!
This page will give you a whirlwind tour of how to work with models.

Before we launch into this, it's worth starting by talking about how Wren stores parameters.
At Wren's core are two types: [`VarName`](@extref) and [`VarNamedTuple`](@extref).

`VarName` is a representation of a variable or part of it, for example: `x`, `x[1]`, or `x.a`.

```@example models
using Wren: @vn

vn1 = @vn(x)     # a variable named `x`
vn2 = @vn(x[1])  # the first element of a variable named `x`
(vn1, vn2)
```

A `VarNamedTuple` is a mapping of `VarName`s to values, which has a Dict-like interface.
However, it is designed to be much more general and performant than a simple `Dict` or `NamedTuple`.

!!! note

    For the full story behind the design of `VarNamedTuple`, please see [the VarNames.jl docs](https://juliabayes.org/VarNames.jl/stable/vnt/motivation/).

## Drawing samples

Calling `rand(model)` gives you a _trace_, i.e., a collection of variable names and values.
Given the above discussion you should not be surprised to learn that the trace is a `VarNamedTuple`!

```@example models
using Wren, Distributions, LinearAlgebra

@wren function f(y)
    x ~ Normal()
    @observe y ~ Normal(x + 1)
    z ~ MvNormal(zeros(3), I)
    return :hello
end
model = f(1.0)

trace = rand(model)
```

To access the values inside the trace, you can index into the `VarNamedTuple` with a `VarName`.

```@example models
trace[@vn(x)]
```

This works even if you only want one part of a variable.
For example, `z` is a vector, but we can access `z[1]` by constructing the appropriate `VarName`:

```@example models
trace[@vn(z[1])]
```

## Calculating log-densities

Once you have a `VarNamedTuple` trace (either from `rand`, or from another source), you can feed it back into the model to calculate the log-densities:

```@example models
logjoint(model, trace)
```

Wren also provides [`logprior`](@ref) and [`loglikelihood`](@ref).

In the workflow we've done, we had to evaluate the model twice: once to generate the trace, and once to calculate the log-density.
This seems a bit wasteful: can't we do both at once?
Indeed, Wren gives you a function for this:

```@example models
accs = rand_with_logp(model)
```

which returns a new object, an [`AccumulatorTuple`](@ref).
You can access the trace with `parameters(accs)`, and the log-density with `logjoint(accs)`.

For importance sampling with the model prior as the proposal, the prior cancels from the
target-to-proposal density ratio. The unnormalised log weight is the log-likelihood:

```@example models
using Random: Xoshiro

proposal = rand_with_logp(Xoshiro(468), model)
proposal_params = parameters(proposal)
logweight = loglikelihood(proposal)
```

This draws the sample and computes its log weight in one model evaluation.
For a different proposal, include its log density in the density ratio.

## General evaluation

At this point, you are probably wondering what an `AccumulatorTuple` is.
This is a good point to talk about the general evaluation framework that Wren uses, which is heavily lifted from DynamicPPL.

The entry point for _all_ Wren model evaluation is the [`WrenCore.evaluate`](@ref) function.
Its first four arguments are:

  - a *model* (of course!);
  - a *latent source* (a specification of how to obtain the values of latent variables);
  - a *parameter space* (a specification of how variables should be transformed); and
  - a set of *accumulators* (the outputs of the model which we are interested in).

!!! note

    The latter three arguments *directly* map to DynamicPPL's initialisation strategies, transform strategies, and accumulators. If you want to read more about that, please see [the DynamicPPL docs](https://turinglang.org/DynamicPPL.jl/stable/evaluation/). (All of that is written by me as well!)

The parameter space argument is slightly more advanced, and we'll leave it for later.
For now, suffice it to say that we'll always use [`WrenCore.OriginalSpace()`](@ref) as this value, which essentially stipulates that all variables should not be transformed.

Let's focus on the other two arguments: the latent source and the accumulators.

A latent source says how to obtain the value of, say, `x` when evaluating the model.
We can either say: 'I don't have one! Please draw a fresh sample of `x` from the prior':

```@example models
using Random: Xoshiro
FromPrior(Xoshiro(468))
```

or we can say, 'I have one stored in this `VarNamedTuple`! Please use it':

```@example models
FromValues(trace)
```

An accumulator, on the other hand, is a struct that holds information that is collected during model evaluation.
An `AccumulatorTuple` wraps multiple such objects (with some special magic, which is mostly hidden, to maximise type stability).
Because most users shouldn't really need to reach for these types on a daily basis, we keep them in WrenCore rather than re-exporting them from Wren.

```@example models
using WrenCore:
    AccumulatorTuple, LogPriorAccumulator, LogLikelihoodAccumulator, ParamsAccumulator

accs =
    AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator(), ParamsAccumulator())
```

The good thing about this design is that you can choose which accumulators to include: for example, if you aren't interested in the log-density, you can include _only_ the `ParamsAccumulator`, and the log-density calculations will be skipped entirely.

Now that we have the ingredients, let's put it all together:

```@example models
using WrenCore: evaluate, OriginalSpace, EvalContext

retval, accs =
    evaluate(model, FromValues(trace), OriginalSpace(), accs, EvalContext{false}());
```

The last argument, [`EvalContext`](@ref), controls whether model arguments are copied during evaluation.
This is only needed when differentiating with Enzyme, so you should almost always pass `EvalContext{false}()` here.

`evaluate` returns a tuple of the model's return value:

```@example models
retval
```

along with a new `AccumulatorTuple` containing the accumulated values:

```@example models
accs
```

As hinted at before, Wren then provides a few convenience functions to access the accumulated values, namely [`logjoint`](@ref), [`logprior`](@ref), [`loglikelihood`](@ref), and [`parameters`](@ref).
Please see their respective docs for more information!

## Conditioning

To condition a model on observed data, pass a `NamedTuple` to the `|` operator:

```@example models
observed = (y=1.0,)

@wren function f_orig()
    x ~ Normal()
    y ~ Normal(x + 1)
end

cond_model = f_orig() | observed
```

The equivalent function is [`condition`](@ref):

```@example models
cond_model = condition(f_orig(), observed)
```

Each key names a whole variable. For example, `(y=[1.0, 2.0],)` observes every
`y[i] ~ dist` statement. Partial conditioning of an array or field is not supported.
A `missing` entry does not make that entry latent.
To mix observations and latent values, use separate variables or condition whole
variables in a submodel.

Conditioning replaces the entire value of each supplied key.
To remove conditioning, use `decondition(cond_model, :y)`, or
`decondition(cond_model)` to remove all conditioning.
See [Conditioning submodels](@ref submodel-conditioning) for nested observations
and deconditioning variables inside child models.
Latent parameters remain `VarNamedTuple`s, as described above.

## [Vectorisation](@id vectorisation)

Wren models can be converted into a vectorised form, i.e., one where the parameters are supplied as a vector rather than a `VarNamedTuple`.
This is done with the `VecModel` constructor, which takes a model and an `AbstractADType` object:

```@example models
using Enzyme

vecmodel = VecModel(model, AutoEnzyme(; mode=Reverse))
```

If you are only interested in vectorisation and not AD, you can pass `nothing` as the second argument:

```@example models
noad_vecmodel = VecModel(model, nothing)
```

This is mostly useful when you want to handoff a Wren model to a different library which expects a vector interface.
In particular, `VecModel`s obey the [LogDensityProblems.jl interface](https://github.com/tpapp/LogDensityProblems.jl), and thus can be used with a wide range of other Julia packages that expect this.

You can draw a vectorised sample from a `VecModel` with the `rand` function:

```@example models
v = rand(vecmodel)
```

or you can convert existing parameters into a vectorised form with [`vectorise_latents`](@ref):

```@example models
ps = FromValues(rand(model))         # This is a VarNamedTuple in original model space.
v = vectorise_latents(ps, vecmodel)  # Convert it to vectorised (and unconstrained) form.
```

Then you can calculate the log-density of these vectorised parameters, and their gradient:

```@example models
import LogDensityProblems as LDP

LDP.logdensity(vecmodel, v)               # Works without an adtype.
LDP.logdensity_and_gradient(vecmodel, v)  # Requires an adtype.
```

!!! note

    If you have a use for higher-order derivatives like Hessians, please open an issue!

Wren in fact exports shorter functions for these, namely [`lp`](@ref) and [`lp_and_grad`](@ref), which are directly equivalent to the above.
Additionally, Wren provides an in-place gradient [`lp_and_grad!`](@ref) function, which may be useful for squeezing every last bit of performance out.
LogDensityProblems.jl does not provide this (but if you have a good usecase for it, please consider [adding a comment on this issue](https://github.com/tpapp/LogDensityProblems.jl/issues/115)!).

None of these functions copy the parameter vector, so the model must not write into its latent variables; see the "Mutability" section of the syntax tutorial.

By default, `VecModel`s are always run in [`UnconstrainedSpace()`](@ref): that is, all parameters are transformed into a form where they can take any value in ℝ^n.
The log-density calculation will then include the log-Jacobian term corresponding to this transformation.
If this is not desired, you can specify a different parameter space, or a different log-density calculation function as a keyword argument to the [`VecModel`](@ref) constructor: please see its docstring for more info.

### Collecting outputs in one evaluation

Use [`evaluate`](@ref) with a `VecModel` to obtain the density components together at a
parameter vector. This reuses its prepared vector mapping:

```@example models
retval, components = evaluate(noad_vecmodel, v)
(logprior(components), loglikelihood(components), WrenCore.logjac(components))
```

`logprior` always uses original model coordinates. `logjac` is the log-absolute determinant
for the forward transformation from original coordinates to the vector's parameter space.
The joint density in that space is therefore:

```@example models
logprior(components) + loglikelihood(components) - WrenCore.logjac(components) ≈
lp(noad_vecmodel, v)
```

In `OriginalSpace()`, `logjac` is zero. These coordinate transforms do not, in general,
turn the prior into a standard normal distribution.

Select other outputs by supplying accumulators. For example, collect the parameters and
likelihood without collecting the prior or Jacobian:

```@example models
selected = AccumulatorTuple(ParamsAccumulator(), LogLikelihoodAccumulator())
retval, selected = evaluate(noad_vecmodel, v, selected)
(parameters(selected), loglikelihood(selected))
```

Supplied accumulators are not copied or reset and may be updated in place. Use fresh
accumulators for independent evaluations. As with `lp`, the parameter vector is not copied,
so the model must not write into its latent variables.

### Evaluating a batch of parameter points

With Reactant.jl loaded, [`prepare_batch`](@ref) compiles evaluations across matrix columns:

```julia
using Reactant
using WrenCore: prepare_batch
using Wren: Dists

@wren function count_model(count)
    rate ~ Dists.Exponential()
    @observe count ~ Dists.Poisson(rate)
end

batch_model = VecModel(count_model(3), nothing)
X = reshape(log.([0.5, 1.5, 3.0]), 1, :)
batch = prepare_batch(batch_model, X)
components = Array(batch(X))
```

The three result rows contain the original-space log-prior, log-likelihood, and forward
log-Jacobian. Each column corresponds to the same input column. Select other outputs with
a function of the accumulators:

```julia
likelihood_batch = prepare_batch(batch_model, X, loglikelihood)
loglikelihoods = vec(Array(likelihood_batch(X)))
```

The output function may return a number or numeric vector. Accumulators can also be
selected with the `accs` keyword, as for single-point evaluation.
Use the distributions in `Wren.Dists` for compatibility with Reactant's traced numbers.

Reuse the prepared callable with matrices of the same size. Inputs and results can stay
on the Reactant device:

```julia
X_device = Reactant.to_rarray(X)
components_device = batch(X_device)
```

Preparation includes compilation. Large batches can benefit from tensor operations across
points; small batches can be faster with ordinary Julia evaluation. The prepared callable
owns its model arguments and isolates mutable arguments and accumulators between points.
The model must keep its latent variables and their dimensions fixed, as for `VecModel`.
Branch values must have compatible types, and loop lengths must be known during preparation.
Called Julia functions and inline-created submodels follow Reactant's tracing rules.
The scalar LogDensityProblems and gradient interfaces continue to use `VecModel`.

!!! warning

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
