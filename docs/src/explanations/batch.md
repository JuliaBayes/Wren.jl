# `@batch` in detail

## How the macro works

TODO... (this part will likely change in the short-term future; I don't want to write it until I've figured out dim-aware `Indexed`).

## Providing your own overload

!!! note

    In general, overloads of `WrenCore.batched` for 'typical' distributions should only really exist in WrenCore itself.
    The reason for this is:

      - if you have an implementation of `batched` for a distribution which others might want to use, then it should be in `WrenCore` so that everyone else can use it.
        **We'd very much appreciate help with implementing some of these!**

      - if you have an implementation of `batched` for a distribution which is very specific to your own code and not worth sharing, you are almost certainly better off just inlining the log-density and using `@addloglikelihood` instead of `@batch`

    The only case where you should really have your own overload is if your package defines a custom distribution type (that isn't in itself worth upstreaming).

With that out of the way, here are a good series of steps for implementing your own overload of `batched`.
This workflow is quite LLM-friendly and so you can point your favourite model at this docs page.

 1. Come up with several different possible implementations of `logdensityof`.
    Usually it makes sense to test a 'loop' version (one where the log-density for each element is summed in a loop) and a 'broadcasted' version which just uses Julia's broadcasting.

 2. Define the variants in a file in `bench/batch/gpu`. (The `gpu` name is a little bit of a misnomer)
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
