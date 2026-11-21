# `@batch` in detail

As described in [the Syntax page](@ref batched), the `@batch` macro is a way for Wren to apply automatic vectorisation to tilde-statements.
For many standard cases this will Just Work.
However, there are also very many cases where it will refuse to work.
This page will try to explain the reasons behind this design, and how to work around it (if possible).

!!! note "Minibatches"

    It is also intended for the `@batch` macro to be a way to implement minibatches / subsampling in Wren.
    This has not yet been implemented, but should not be overly difficult to add.

    When this is done, it will be slightly unfortunate that the `@batch` macro will have overloaded semantics.
    At the present moment we don't anticipate that this should be a problem.
    However, if this proves to be a blocker for you, please do [open an issue](https://github.com/JuliaBayes/Wren.jl/issues)!
    We can consider how to design this to separate the two use cases.

## Why batch?

To begin, let's start by making the case _for_ vectorisation in general.
In other words, why should we go through all the trouble of implementing vectorisation even when it doesn't always work?

Here are a few motivations:

 1. We can perform (some) detection of common subexpressions.
    One of the simplest examples of this is:

    ```@example batch
    using WrenDists, Chairmarks, LinearAlgebra, Statistics
    
    function f_loop(y, mu, sigma)
        s = 0.0
        for i in eachindex(y, mu)
            s += logdensityof(Normal(mu[i], sigma), y[i])
        end
    end
    
    function f_vect(y, mu, sigma)
        return logdensityof(MvNormalScale(mu, sigma * I), y)
    end
    
    function compare(N)
        y, mu, sigma = randn(N), randn(N), 1.0
        loop = median(@be f_loop($y, $mu, $sigma)).time
        println("loop: $loop")
        vect = median(@be f_vect($y, $mu, $sigma)).time
        print("vect: $vect")
    end
    
    compare(10_000)
    ```

    Here, the `MvNormalScale` distribution avoids re-calculating `log(sigma)` for each element of the vector, and is therefore substantially faster than the loop version.

 2. Sometimes, even without common subexpression elimination, vectorised operations are faster than loops.
    In the following example, Wren does not perform any common subexpression elimination, but the vectorised version is still faster than the loop version:

    ```@example batch
    using WrenCore, WrenDists, Enzyme
    
    N = 10_000
    y = rand(1:5, N);
    lambda = zeros(N);
    
    @wren function f_poisson_loop(y, lambda)
        for i in eachindex(y)
            lambda[i] ~ Exponential(1.0)
            @observe y[i] ~ Poisson(lambda[i])
        end
    end
    
    @bmodel f_poisson_loop(y, lambda) AutoEnzyme(; mode=Reverse)
    ```

    ```@example batch
    @wren function f_poisson_vect(y, lambda)
        @batch for i in eachindex(y)
            lambda[i] ~ Exponential(1.0)
            @observe y[i] ~ Poisson(lambda[i])
        end
    end
    
    @bmodel f_poisson_vect(y, lambda) AutoEnzyme(; mode=Reverse)
    ```

 3. Finally, a large part of the motivation for vectorisation is that it generates code that is more friendly for Reactant.

    In general, naive `for` loops are problematic for Reactant because they are _unrolled_, i.e., Reactant generates XLA code that has one copy of the loop body for each iteration of the loop.
    This can lead to massive blow-ups in compilation time.
    The general way around this is to either vectorise the code yourself, or use Reactant's `@trace for ... end` macro, which lowers instead to StableHLO control flow operations.
    Unfortunately, the implementation of `@trace` generates a closure inside the model body, which is disallowed by RuntimeGeneratedFunctions.jl, and so cannot be used in Wren models.

!!! note

    It's okay to use `@trace` in the definition of a function that Wren calls; you just can't use `@trace` at top-level in the model body itself.

    Thus, part of the intent of `@batch` is to abstract away this problem from users: instead of having to figure out `@trace` and all its friends, users can just write `@batch` and have it work (ideally).

It's worth noting that a skilled user could of course implement all of these themselves.
In fact, a skilled user could simply define a log-density function themselves and circumvent the PPL entirely!
However, the idea is that `@batch` allows the user to express their model in a natural way and get the optimisation 'for free'.

In summary, `@batch` is a way for Wren to provide increased expressivity without sacrificing performance.

## How does Wren perform batching?

Let's start with the simplest example, the `Normal` case above where the standard deviation is a constant.
Wren's job is to turn the expression

```julia
y[i] ~ Normal(x[i], sigma)
```

into something that's equivalent to

```julia
y[1:N] ~ MvNormalScale(x[1:N], sigma * I)
```

(Wren also tries to use as many `@view`s as possible to avoid allocating new slices, but that's not important for the present discussion.)

Note that this transformation happens at macro time, i.e., we don't actually know what `Normal` means; we just know that it's some function called `Normal`.
Thus, we can't pattern-match on the `Normal` distribution itself.
We can, however, pass it into a function.

The design looks something like this:

```julia
# Direct, mechanical, replacement of `i` with `1:N`.
new_lhs = replace_i_with_1toN(lhs)

# Split the RHS into its function and arguments.
rhs_func, rhs_args = split(rhs)

# Replace `i` with `1:N` in the arguments.
# i.e. x[i], sigma -> x[1:N], sigma
new_rhs_args = map(replace_i_with_1toN, rhs_args)

# Put it back together.
new_rhs = :(batched($rhs_func, $new_rhs_args...))

# Emit the new tilde-statement.
emit(:(new_lhs ~ new_rhs))
```

The resulting tilde-statement will thus look like `y[1:N] ~ batched(Normal, x[1:N], sigma)`.
Now, we haven't yet defined what `batched` actually does: this has to be done at runtime.
In order to make that work, we would have to provide an overload of `batched` for the `Normal` distribution, which to a first approximation might look something like this:

```julia
# Our first try.
function batched(::Type{Normal}, mu::AbstractVector, sigma::Number)
    return MvNormalScale(mu, sigma * I)
end
```

This at least captures the spirit of what we're trying to do, but fails for a couple of reasons.
One is that `batched(Normal, ...)` can't be dispatched on, because `Normal` is a type constructor, and will have type `DataType`.
To avoid this, we need to modify the macro to put `Normal` inside a `Val`, which will let us dispatch on it.

```julia
# Our second try.
function batched(::Val{Normal}, mu::AbstractVector, sigma::Number)
    return MvNormalScale(mu, sigma * I)
end
```

The second problem is that we need to have a fallback for generic functions that we don't know how to batch.
For example, if the user writes

```julia
f(mu, sigma) = Normal(mu, sigma)

@batch for i in 1:N
    y[i] ~ f(x[i], sigma)
end
```

this is actually the same as before, but Wren will lower this to `batched(Val{f}(), ...)` which won't have a method defined for it (unless the user also kindly writes that method; but that defeats the purpose).

To handle this, we have a fallback method for `batched` which effectively does the 'reverse batching' and then combines the distributions with `Wren.independent` (which is semantically equivalent to `Distributions.product_distribution`, if you're more familiar with that):

```julia
function batched(iters, ::Val{F}, args...) where {F}
    return Wren.independent([F([_unindex(arg, i) for arg in args]...) for i in iters])

    # In our example above, this will be equivalent to
    # Wren.independent(Normal(x[1:N][i], sigma) for i in 1:N)
end
```

!!! note

    It's worth pointing out that `independent` is _not_ a vectorised computation.
    Even though it looks like a one-liner, it just hides the loop better: the calculation of the log-density is still done by iterating over the array of distributions stored inside.

Here, `args` have already been batched by the macro, so `args[1]` would be a vector (`mu`) and `args[2]` a scalar (`sigma`).
For this to work correctly, we need to modify our setup with a couple of things:

  - We need to pass in `iters` into `batched`.

  - We need a function `_unindex` which 'reverts' the batching.
    Note that this is all done at runtime and `x[1:N]` has already been executed and stored as a vector (or view thereof).
    We don't have the original `Expr` to work with, and we can't just replace `1:N` with `i` any more; we have to undo the batching by indexing into the vector.

    This in turn means that we need to remember which arguments were batched!
    Wren achieves this by making the macro wrap each batched argument in a `WrenCore.Indexed` struct, so `args[1]` is _really_ `Indexed(x[1:N])`, and `args[2]` is just a plain `sigma`.

Putting this all together, that means that the correct overload for `batched` is as follows:

```julia
# Our third (and correct) try.
function batched(iters, ::Val{Normal}, mu::Indexed{D,N,<:AbstractVector}, sigma::Number)
    return MvNormalScale(mu, sigma * I)
end
```

## Multiple dimensions

The only things above that remain unexplained are the type parameters `D` and `N` in the `Indexed` type.
These type parameters are unnecessary as long as `Indexed` only wraps an `AbstractVector`, as was the case in the example above.
In such cases, we know that the vector dimension corresponds to the batch, and we can trivially un-batch it by indexing into the vector.

However, this case is more difficult.
Here, `probs` is an `N * K` matrix where `N` is the batch dimension and `K` is the number of categories:

```julia
@batch for i in 1:N
    y[i] ~ Categorical(probs[i, :])
end
```

In order to provide a `batched` implementation for Categorical, Wren needs to know which dimension of `probs` corresponds to the batch dimension.
It does so by recognising that `i` is the first index of `probs`, and so the batch dimension is the first dimension.
The `@batch` macro will then lower the above to

```julia
y[1:N] ~ batched(1:N, Val{Categorical}(), Indexed{1,2}(probs[1:N, :]))
```

where `1` is the batch dimension and `2` is the number of dimensions of `probs`.

Our `batched` implementation for Categorical can then use this information:

```julia
function batched(iters, ::Val{Categorical}, probs::Indexed{1,2,<:AbstractMatrix})
    return BatchedCategorical(probs)
end
```

Conversely, if the user instead wrote `y[i] ~ Categorical(probs[:, i])`, then the macro would lower to

```julia
y[1:N] ~ batched(1:N, Val{Categorical}(), Indexed{2,2}(probs[:, 1:N]))
```

and we can use this information in our overload to transpose the matrix before passing it to `BatchedCategorical`:

```julia
function batched(iters, ::Val{Categorical}, probs::Indexed{2,2,<:AbstractMatrix})
    return BatchedCategorical(permutedims(probs))
end
```

## Limitations

Although the above already seems quite comprehensive, it unfortunately does not take into account many other syntactic possibilities that can mess up the outputs.

### Macro-time failures

In particular, applying a function to the batched argument can have arbitrary effects and thus we cannot safely broadcast it.
Consider this fairly common example:

```julia
@batch for i in 1:N
    y[i] ~ Poisson(exp(x[i]))
end
```

Here, we need to batch the expression `:(exp(x[i]))`.
Simply replacing `i` with `1:N` will give us `exp(x[1:N])`, but that's not what we want: we want `exp.(x[1:N])` instead.

We _could_ try to detect this case and insert the dot (and indeed an early version of Wren did exactly this), but unfortunately this is only a safe transformation if the function maps scalars to scalars.
Consider this alternative:

```julia
@batch for i in 1:N
    y[i] ~ Normal(sum(x[i, :]))
end
```

We can't expand this to `y[1:N] ~ batched(1:N, Val{Normal}(), sum.(x[1:N, :]))`, because that will have the wrong behaviour.
We really want to do something like

```julia
y[1:N] ~ batched(1:N, Val{Normal}(), sum(x[1:N, :], dims=2))
```

or to be precise, since `batched` can't tell upfront that `sum` can take a `dims` argument, we can use the more generic `mapslices` function:

```julia
y[1:N] ~ batched(1:N, Val{Normal}(), mapslices(sum, x[1:N, :], dims=2))
```

This would have the correct semantics, but unfortunately Wren has zero way of knowing which of the two situations we are in!
So, in general, Wren will refuse to batch any arguments that are not just indexing expressions.
In such cases, Wren will just issue an info message saying that it is falling back to `independent`.
This won't affect correctness, but might be slower due to the lack of vectorisation.

The way to solve this is to lift the function call outside the loop:

```julia
sums = sum(x, dims=2)
@batch for i in 1:N
    y[i] ~ Normal(sums[i])
end
```

Or equivalently, for the Poisson case, Wren also provides `PoissonLog` which allows you to pass `log(lambda)` directly, so you can write

```julia
@batch for i in 1:N
    y[i] ~ PoissonLog(x[i]) # instead of Poisson(exp(x[i]))
end
```

### Runtime errors

Sadly, there are also some pathological cases where even this is not enough!

To be specific, there is also a failure mode where the fallback for `batched` cannot determine how to un-batch the arguments with certainty.
By the time we hit this method, we are already executing the model, and (by definition) we can't fall back to `independent`.
Thus, the only safe thing to do is to throw an error.

This can happen if there is an indexing expression which:

  - includes the loop variable;
  - extracts a single row along a different dimension;
  - and results in an array with 2 or more dimensions.

A concrete example of this is with a 3-D array `s` which has shape `(A, N, B)` and a loop variable `i` which is the first index:

```julia
@batch for i in 1:N
    y[i] ~ SomeDist(s[:, i, 5])
end
```

This will call `batched(1:N, Val{SomeDist}(), Indexed{2,3}(s[:, 1:N, 5]))`.
Assuming we don't have a custom method for this, we will fall back to the default `batched` implementation which will try to un-batch the argument by indexing into it.

The wrapped data (`s[:, 1:N, 5]`) is an `A * N` matrix.
To un-batch it we would need to know to index along the _second_ dimension of this matrix.
You might think that we can infer this from the `Indexed{2,3}` type (and in particular the `2`), but unfortunately this is not enough information.
Consider this case:

```julia
@batch for i in 1:N
    y[i] ~ SomeDist(s[5, i, :])
end
```

Here, we _also_ have an `Indexed{2,3}(...)`, but now the batch dimension is the _first_ dimension of the wrapped matrix `s[5, 1:N, :]`.
The only thing that changed was that we swapped which dimension was collapsed (with the index `5` and which dimension was left intact (with the `:`).

In general, the value `2` in the `Indexed{2,3}` type only tells us that the second dimension of the **original** array was batched, but it doesn't tell us which dimension of the **resulting** array corresponds to the batch dimension.

We can only safely determine the latter when:

  - the resulting array has only one dimension (i.e., it is a vector);
  - the resulting array and the original array have the same number of dimensions.
    In this case the batch dimension is just the first type parameter of `Indexed` (i.e., the `2` in `Indexed{2,3}`).

When we can't determine this, Wren will throw an error.
Thankfully, these cases are quite pathological and should be easy to avoid.
In particular, the above example can be rewritten as

```julia
s5 = s[5, :, :]
@batch for i in 1:N
    y[i] ~ SomeDist(s5[i, :])
end
```

which will work just fine.

## Implementing batched log-densities

The above discussion should show you how to implement `batched` and get it to dispatch to your custom batched-distribution struct.
However, there's then also a question of how the log-density for that struct should be implemented.

!!! note

    In general, overloads of `WrenCore.batched` for 'typical' distributions should only really exist in WrenCore itself.
    The reason for this is:

      - if you have an implementation of `batched` for a distribution which others might want to use, then it should be in `WrenCore` so that everyone else can use it.
        **We'd very much appreciate help with implementing some of these!**

      - if you have an implementation of `batched` for a distribution which is very specific to your own code and not worth sharing, you are almost certainly better off just inlining the log-density and using `@addloglikelihood` instead of `@batch`

    The only case where you should really have your own overload is if your package defines a custom distribution type (that isn't in itself worth upstreaming).

Here is a good series of steps for implementing your own overload of `batched`.
This workflow is quite LLM-friendly and so you can point your favourite model at this docs page.

 1. Come up with several different possible implementations of `logdensityof`.
    Usually it makes sense to test a 'loop' version (one where the log-density for each element is summed in a loop) and a 'broadcasted' version which just uses Julia's broadcasting.

 2. Define the variants in a file in `bench/batch/gpu`.
    (The `gpu` name is a little bit of a misnomer.)
    The format for this can be seen in the other folders in that directory.
    Start by including `common.jl`.
    In those files, `ScalarIndexed` can be used to wrap a function that performs scalar indexing in `@allowscalar` (otherwise Reactant will complain).

 3. Then finally call `run_variants`, and run the file with the environment in `bench`.
    This should print out a nice table of timings for each variant at different sizes, allowing you to choose between them.
    (This can take a while, go make a cup of tea / coffee :))

In general it would also be ideal to repeat these benchmarks on GPU.
However based on previous experience there tends to not be very much difference between different implementations on GPU, so often it will be the CPU benchmarks that help you decide which implementation is best.

Once you've done this, you will often find that there isn't a single implementation that is optimal across all cases.

### Reactant vs non-Reactant

If you find that one implementation is better without Reactant (i.e., when run as pure Julia code, or differentiated with Enzyme), but a different implementation is better under Reactant, then you can use `ReactantCore.within_compile` to select the best implementation for each case.
`ReactantCore` is already a hard dependency of WrenDists, so you can just use it directly:

```julia
function DensityInterface.logdensityof(mybatcheddist, x)
    if within_compile()
        my_reactant_logdensityof(mybatcheddist, x)
    else
        my_julia_logdensityof(mybatcheddist, x)
    end
end
```

This will allow you to have different implementations for the Reactant and non-Reactant cases.
Indeed often the broadcasted version is better for Reactant, while the loop version is better for non-Reactant.

### Primal vs Enzyme gradient

In the unlikely case that one implementation is better for the primal and the other is better for the Enzyme gradient, you can also define an Enzyme rule to optimise the gradient.
You can see `BatchedLogNormal` for an example of this.
In this case, please make sure to also add a test for the Enzyme rule [using `EnzymeTestUtils`](https://enzymead.github.io/Enzyme.jl/stable/generated/custom_rule/#Testing-our-rules) to guarantee correctness!

Note that Enzyme rules are not currently picked up by Reactant, so there is no way to separately optimise the Reactant gradient and Reactant primal.
