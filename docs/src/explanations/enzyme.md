# [Working with Enzyme](@id enzyme)

The Julia ecosystem has had (infamously) [a number of automatic differentiation packages](https://juliadiff.org/), each with their own tradeoffs.
Wren adopts a _partly_ opinionated approach to AD.
All (sensible) AD backends are supported via [DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface), so the user can choose their preferred backend.
However, at the same time, Wren also tries very hard to make it possible for you to use reverse-mode Enzyme as _the_ backend.

The developer(s) of Wren are not affiliated with any AD backend and this opinion is not rooted in any politics or personal disagreement.
It is purely based on the fact that reverse Enzyme is the fastest AD backend across a wide range of problems (see, e.g., [the benchmarks on Turing.jl](https://juliabayes.org/ADTests/)).
Thus, we consider it a disservice to choose any other backend as the default, as that simply leaves performance on the table.

At the same time, we recognise that Enzyme can be difficult to use and still has a number of edge cases.
Wren does its best to make this experience as seamless as possible for users.
However, it is **greatly** appreciated if you [help us by reporting cases](https://github.com/JuliaBayes/Wren.jl/issues) where Enzyme fails to work.
We can (usually) help with minimising bugs and filing issues upstream, which will not only help you but also everyone else who uses Enzyme!

## Runtime activity

Wren does not _generally_ require you to enable runtime activity when using Enzyme.
Thus, you can simply use `DEFAULT_AD` (an alias for `AutoEnzyme(; mode=Reverse)`) as the adtype when e.g. [sampling with NUTS](@ref nuts), or [constructing a `VecModel`](@ref vectorisation).

However, depending on how the model is written, sometimes Enzyme detects that it is necessary to use runtime activity.
If you are familiar with Enzyme, you will know that this usually causes an error to be thrown.
**However, Wren will, by default, detect such situations and automatically enable runtime activity for you**.
Note that even with runtime activity enabled, Enzyme still tends to be the fastest AD backend, so you do not really need to be concerned about this.

This behaviour can be controlled via the function `WrenCore.set_auto_runtime_activity!`:

```@docs
WrenCore.set_auto_runtime_activity!
```

### When exactly is runtime activity required?

The easiest way to trigger a need for runtime activity is to bundle differentiable and non-differentiable data into the same struct.
Often there is not much you can do as a user to avoid this situation, which is precisely why Wren intentionally catches the error for you.

Differentiable data, in the context of Wren, refers to the actual parameters of the model (i.e., the things we want to know gradients with respect to).
For example, if you have `x ~ MvNormal(...)`, then `x` contains differentiable data during model evaluation.
Non-differentiable data usually means constants that are supplied to the model.

Consider, for example, the following model:

```@example enzyme-rta
using Wren, Wren.Dists, Enzyme

@wren function f_rta(sigma, y)
    m ~ MvNormalScale(zeros(4), I)
    y ~ MvNormalScale(m, Diagonal(sigma))
end
```

Here, `m` is differentiable data, whereas `sigma` is non-differentiable data.
Mixing these two things in the same struct (i.e., `MvNormalScale`) will trigger Enzyme's runtime activity requirement.

However, at the same time, it's clearly not possible to rewrite the model to avoid this.
If the offending struct is only used as an observation distribution, then you can [manually write out the log-likelihood and use `@addloglikelihood`](@ref addloglike) and thus avoid making the struct, but this is less expressive and will make your model less usable.

### Avoiding runtime activity by copying

The need for runtime activity in the above case can also be overcome by creating a copy of `sigma` inside the model.

```@example enzyme-rta
@wren function f_copy(sigma, y)
    m ~ MvNormalScale(zeros(4), I)
    y ~ MvNormalScale(m, Diagonal(copy(sigma)))
end
```

We can prove that this doesn't need runtime activity by setting `WrenCore.set_auto_runtime_activity!(false)` (which will error if runtime activity is required):

```@example enzyme-rta
sigma, y = rand(4), randn(4)

WrenCore.set_auto_runtime_activity!(false)
VecModel(f_copy(sigma, y), DEFAULT_AD)
```

You can also see that the second type parameter of `EnzymeCore.ReverseMode` is `false`, which indicates that runtime activity was not enabled.

Conversely, the original model (without the copy) will require runtime activity:

```@example enzyme-rta
# Reenable this, or else it'll error!
WrenCore.set_auto_runtime_activity!(true)

VecModel(f_rta(sigma, y), DEFAULT_AD)
```

The second type parameter is now `true`.

Is the copy or the runtime activity the better solution?
We can [benchmark](@ref benchmarking) to see which is faster:

```@example enzyme-rta
sigma, y = rand(4), randn(4)

@bmodel f_copy(sigma, y) DEFAULT_AD
```

```@example enzyme-rta
@bmodel f_rta(sigma, y) DEFAULT_AD
```

In this case we can see that using the copy slows things down.
In general, the cost of copies will scale with the size of the arguments and can in principle get very large, whereas runtime activity roughly scales with the number of pointer operations that Enzyme is doing.
Furthermore, eliding copies also makes the primal faster (as seen above), and also reduces GC pressure.

## [Copying model arguments on evaluation](@id enzyme-copyargs)

Wren sometimes copies model arguments when evaluating models with Enzyme.

### Why do we need copies?

When originally written, Wren would never copy any model arguments during model evaluation (it would copy them when _creating_ the model, but not when _evaluating_ it, since the latter is usually done many more times).

This very simple rule had to (sadly) be changed because it could silently cause incorrect results with Enzyme.
Specifically, this could happen if differentiable data was written into a model argument, and then read out again.

Consider the following model, which is only slightly modified from the previous section's to get rid of `sigma`:

```@example enzyme-copyargs
using Wren, Wren.Dists, Enzyme

@wren function no_sigma(m, y)
    m[1:4] ~ MvNormalScale(zeros(4), I)
    @observe y[1:4] ~ MvNormalScale(m[1:4], I)
end
```

The astute reader will notice that this is actually [a batched for-loop in disguise](@ref batched):

```@example enzyme-copyargs
@wren function batched(m, y)
    @batch for i in eachindex(y)
        m[i] ~ Normal(0, 1)
        @observe y[i] ~ Normal(m[i], 1)
    end
end

model = batched(randn(4), randn(4))
```

Anyway, Wren emits a model evaluation function that conceptually looks like:

```julia
function batched_eval(params, m, y)
    lp = 0.0
    m[1:4] = params[1:4]
    lp += logpdf(MvNormal(zeros(4), I), m[1:4])
    lp += logpdf(MvNormal(m[1:4], I), y[1:4])
    return lp
end
```

Notice that here `params` is differentiable, but `m` and `y` are not.
Wren marks the latter two (the model arguments) with `DI.Constant`, which then translates into `Enzyme.Const`, so the invocation of AD is morally equivalent to

```julia
Enzyme.gradient(Reverse, batched_eval, params, Const(m), Const(y))
```

Since the log-likelihood term (`y[1:4] ~ ...`) depends on `m[1:4]`, which in turn is derived from `params[1:4]`, the log-likelihood should contribute to the gradient with respect to `params`.

However, this is unfortunately lost when we write the differentiable data from `params` into the non-differentiable `m`:

```julia
m[1:4] = params[1:4]
```

Since `m` was declared as `Const`, it doesn't have a 'shadow': in other words, it doesn't have a corresponding array to store its gradients in.
Thus, even though `d(m)/d(params)` is nonzero, this fact isn't captured anywhere.
So when we reach the log-likelihood term, Enzyme has no way of 'tracking' the gradient of the log-likelihood with respect to `params`.

This behaviour is intentional in Enzyme, so there isn't an 'upstream fix' available.
Thus, when evaluating models with Enzyme, Wren will sometimes insert calls to `_dealias` at the top of the model evaluation function.
(We detect when models are being run with Enzyme using the `EvalContext` struct, which is passed as an argument to the evaluation function.)

To see which model arguments will be copied when using Enzyme, you can call the internal `WrenCore._copied_args` function:

```@example enzyme-copyargs
WrenCore._copied_args(model)
```

### Eliding copies

Wren attempts to save you from this performance cost by eliding copies where possible.

It does so in two ways.
Firstly, it performs some basic dataflow analysis to try to avoid copying things that definitely don't need to be copied.
Unfortunately, this is very conservative since Wren only has access to the syntax tree.
(In fact it's arguable that even this analysis is kind of pointless given the next point; however, it still exists since it's just a strict improvement over not doing it at all.)

Secondly, and more interestingly, Wren has a `reduce_copies` function which essentially tries eliding each argument's copy one by one, and sees if the gradient (evaluated at a few points) is still correct.

```@example enzyme-copyargs
new_model = WrenCore.reduce_copies(model, DEFAULT_AD)
WrenCore._copied_args(new_model)
```

This costs very little time compared to Enzyme's first compilation, and so in fact when constructing a `VecModel` with Enzyme, this will be done automatically:

```@example enzyme-copyargs
vm = VecModel(model, DEFAULT_AD)
WrenCore._copied_args(vm)
```

It should be noted that this trial-and-error approach is in principle not guaranteed to always be correct.
However, the cases where it will fail are extremely contrived.
Essentially it requires:

  - parameter-dependent control flow, i.e., `if cond(x)` where `x` is itself a random variable;
  - the control flow must have two (or more) different paths, one of which where eliding the copy causes incorrect gradients, and one where it doesn't;
  - `reduce_copies` must be unlucky enough to only hit the latter path during its trial-and-error evaluation.

In general, parameter-dependent control flow already does not work well with vectorisation, so this is pretty unlikely to cause any problems in practice.
However, if desired, the trial-and-error copy elision can be disabled with

```@example enzyme-copyargs
vm = VecModel(model, DEFAULT_AD; reduce_copies=false)
WrenCore._copied_args(vm)
```

or even more simply by just copying the argument yourself inside the model.
If you do this, then Wren's automated analysis will most likely elide Wren's own call to `copy`, but your `copy` will be untouched, which yields exactly the desired result.

## Troubleshooting

### Records with missing fields

For arrays of records with missing fields, give the records a concrete element type:

```julia
Record = @NamedTuple{a::Union{Missing,Float64}, b::Union{Missing,Float64}}
data = Record[(a=1.0, b=missing), (a=missing, b=2.0)]
```

These immutable records contain no references to mutable objects, so Wren can copy the
array without deep copying its elements. The untyped literal
`[(a=1.0, b=missing), (a=missing, b=2.0)]` instead has an abstract `NamedTuple` element
type; Enzyme currently cannot differentiate copies of that representation.
