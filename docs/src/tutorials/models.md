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

## General evaluation

At this point, you are probably wondering what an `AccumulatorTuple` is.
This is a good point to talk about the general evaluation framework that Wren uses, which is heavily lifted from DynamicPPL.

The entry point for _all_ Wren model evaluation is the [`WrenCore.evaluate`](@ref) function.
It takes exactly four arguments:

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
FromParams(trace)
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
using WrenCore: evaluate, OriginalSpace

retval, accs = evaluate(model, FromParams(trace), OriginalSpace(), accs);
```

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

This is in some ways more fundamental, but we had to explain what a `VarNamedTuple` was first. :-)

To condition a model on some observed data, you can use the `|` operator:

```@example models
vnt = VarNamedTuple(y=1.0)

@wren function f_orig()
    x ~ Normal()
    y ~ Normal(x + 1)
end

cond_model = f_orig() | vnt
```

or the equivalent [`condition`](@ref) function:

```@example models
cond_model = condition(f_orig(), vnt)
```

Unfortunately, the `VarNamedTuple`s that you use for conditioning will almost certainly not be the same as those drawn from a model.
Thus, you will likely have to construct a `VarNamedTuple` yourself.
More thorough docs on this will hopefully come at some point in time but here are some syntax examples:

```@example models
# NamedTuple-like syntax.
VarNamedTuple(y=1.0, z=zeros(3))

# From an AbstractDict{VarName}.
VarNamedTuple(Dict(@vn(y) => 1.0, @vn(z.a) => zeros(3)))

# With a macro.
@vnt begin
    # Note the use of `:=` rather than `=`
    y := 1.0
    z.a := zeros(3)

    # This says that `vect[1]` is part of a three-element vector, and that
    # its first value is 1.0. The other values of `vect` are NOT part of
    # the resulting `VarNamedTuple`. This allows you to condition only part
    # of a vector. Note that the template should have the same shape as
    # whatever `vect` in the model will be.
    @template vect = zeros(3)
    vect[1] := 1.0
end
nothing # hide
```

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
ps = FromParams(rand(model))         # This is a VarNamedTuple in original model space.
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

By default, `VecModel`s are always run in [`UnconstrainedSpace()`](@ref): that is, all parameters are transformed into a form where they can take any value in ℝ^n.
The log-density calculation will then include the log-Jacobian term corresponding to this transformation.
If this is not desired, you can specify a different parameter space, or a different log-density calculation function as a keyword argument to the [`VecModel`](@ref) constructor: please see its docstring for more info.

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

## Benchmarking

For the vast majority of users, the easiest way to benchmark will be to use the [`@bmodel`](@ref) macro:

```@example 1
using Wren, Distributions, LinearAlgebra
@wren function f()
    x ~ Normal()
    z ~ MvNormal(zeros(3), abs(x) * I)
end

@bmodel f()
```

```@example 1
using Enzyme
@bmodel f() AutoEnzyme(; mode=Reverse)
```

If this isn't enough and you want to customise _exactly_ what gets benchmarked, and to test correctness, Wren also provides a function [`benchmark_and_test`](@ref) which has a staggering number of options.
(There are also slightly simpler versions, [`benchmark_ad`](@ref), and [`test_ad`](@ref).)
