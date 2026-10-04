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
**However, Wren will, by default, detect such situations and automatically enable runtime activity for you**, issuing a warning in the process.
Note that even with runtime activity enabled, Enzyme still tends to be the fastest AD backend, so you do not really need to be concerned about this message.

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
using Wren, Distributions, Enzyme

@wren function f(sigma, y)
    m ~ MvNormal(zeros(4), I)
    y ~ MvNormal(m, Diagonal(sigma))
end
```

Here, `m` is differentiable data, whereas `sigma` is non-differentiable data.
Mixing these two things in the same struct (i.e., `MvNormal`) will trigger Enzyme's runtime activity requirement.

However, at the same time, it's clearly not possible to rewrite the model to avoid this.
If the offending struct is only used as an observation distribution, then you can [manually write out the log-likelihood and use `@addloglikelihood`](@ref addloglike) and thus avoid making the struct, but this is less expressive and will make your model less usable.

### Copies

The need for runtime activity in the above case can also be overcome by creating a copy of `sigma` inside the model.
In fact, Wren does often copy some model arguments during evaluation in order to ensure correctness (see [the page on argument copying](@ref copies) for more information).
You can see this by printing the model in the REPL:

```@example enzyme-rta
sigma = ones(4);
y = randn(4)
f(sigma, y)
```

Wren shows here that `sigma` is copied at every model evaluation, which in fact circumvents the need for runtime activity.
However, if we were to disable the copying with [`@nocopy`](@ref) annotations, then we would see that runtime activity will be turned on:

```@example enzyme-rta
@wren function f_nocopy(@nocopy(sigma), y)
    m ~ MvNormal(zeros(4), I)
    y ~ MvNormal(m, Diagonal(sigma))
end
VecModel(f_nocopy(sigma, y), DEFAULT_AD)
```

In this printout, the second type parameter of the [`ReverseMode`](https://enzyme.mit.edu/julia/stable/api/#EnzymeCore.ReverseMode) struct is `true`, which indicates that runtime activity has been enabled.

Is the copy or the runtime activity the better solution?
We can [benchmark](@ref benchmarking) to see which is faster:

```@example enzyme-rta
@bmodel f(sigma, y) DEFAULT_AD
# Notice also that runtime activity isn't enabled here.
```

```@example enzyme-rta
@bmodel f_nocopy(sigma, y) DEFAULT_AD
```

In this case we can see that eliding the copy is faster.

If you use [`reduce_copies`](@ref copies) to automatically detect which copies can be safely skipped, it will always opt to remove the copies, even if that requires runtime activity to be enabled.
The reason for this is because the cost of copies scale with the size of the arguments and can in principle get very large, whereas runtime activity scales with the number of loads and stores.
Furthermore, eliding copies also makes the primal faster (as seen above), and also reduces GC pressure.

## [Copying model arguments on evaluation](@id enzyme-copyargs)

!!! warning

    You should not need to read this unless you are actively developing Wren.
    [The tutorial on argument copying](@ref copies) should have enough information for most users of Wren!

When originally written, Wren did not copy any model arguments during model evaluation (it would copy them when _creating_ the model, but not when _evaluating_ it, since the latter is usually done many more times).

This had to be changed because it could silently cause incorrect results with Enzyme.
Specifically, this could happen if differentiable data was written into a model argument, and then read out again.

Consider the following model, which is only slightly modified from the previous section's to get rid of `sigma`:

```julia
@wren function f(m, y)
    m[1:4] ~ MvNormal(zeros(4), I)
    @observe y[1:4] ~ MvNormal(m[1:4], I)
end
```

The astute reader will notice that this is actually [a batched for-loop in disguise](@ref batched):

```julia
@wren function f(m, y)
    @batch for i in 1:4
        m[i] ~ Normal(0, 1)
        @observe y[i] ~ Normal(m[i], 1)
    end
end
```

Anyway, Wren emits a model evaluation function that conceptually looks like:

```julia
function f_eval(params, m, y)
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
Enzyme.gradient(Reverse, f_eval, params, Const(m), Const(y))
```

Since the log-likelihood term (`y[1:4] ~ ...`) depends on `m[1:4]`, which in turn is derived from `params[1:4]`, the log-likelihood should contribute to the gradient with respect to `params`.

However, this is unfortunately lost when we write the differentiable data from `params` into the non-differentiable `m`:

```julia
m[1:4] = params[1:4]
```

Since `m` was declared as `Const`, it doesn't have a 'shadow': in other words, it doesn't have a corresponding array to store its gradients in.
So when we reach the log-likelihood term, Enzyme has no way of 'tracking' the gradient of the log-likelihood with respect to `params`.

This behaviour is intentional in Enzyme and is a result of the `Const` marking.
There isn't a simple way to work around this with DifferentiationInterface, so Wren currently works around this by copying most model arguments during evaluation.
Wren does perform some basic analysis to try to avoid copying things that definitely don't need to be copied, but this is very conservative since Wren only has access to the syntax tree.

The recommended workaround to this is for users to manually use [`@nocopy`](@ref) annotations on model arguments which are safe to not copy, or to use the [`reduce_copies`](@ref) function to automatically detect which arguments can be safely not copied.
