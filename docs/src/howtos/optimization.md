# [Point estimates](@id point-estimates)

Wren provides the `MAP` and `MLE` functions for obtaining point estimates of model parameters (respectively, the maximum a posteriori and maximum likelihood estimates).

This requires you to load [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) which provides an interface to a variety of optimisation algorithms, as well as a package that provides the actual solvers.
In the examples below we use the [limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) solver from [OptimizationOptimJL.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/optim).

!!! note "Only requires WrenCore"

    The extension that provides this functionality is defined in WrenCore, so you don't need to load Wren.
    This means that you don't need to load other MCMC-related dependencies if you just want to use point estimates.

    However, for convenience in the examples below we will use Wren directly.

Note that, depending on which solver you use, you may also need to load an AD backend.
Just as with MCMC, Wren uses reverse-mode Enzyme as its default AD backend.

```@example optimization
using Wren
using Wren.Dists
using OptimizationOptimJL: LBFGS
import Enzyme, Optimization

@wren function poisson_rate(count)
    rate ~ Gamma(2.0, 1.0)
    @observe count ~ Poisson(rate)
end

model = poisson_rate(3)
estim = MAP(model, LBFGS())
```

`MAP` (and `MLE`) returns a [`PointEstimate`](@ref), from which you can access the fitted (model-space) parameters with [`parameters`](@ref).
Here, the MAP estimate of `rate` is approximately 2:

```@example optimization
parameters(estim)
```

For the other fields, you should directly access the `PointEstimate` object's fields.
Briefly:

  - `lp` is the log-density at the optimum;
  - `solution` is the original return value from Optimization.jl.
    You should check the `retcode` to make sure that the optimisation converged successfully.

Likewise, we can use `MLE` to obtain the maximum likelihood estimate of the same model:

```@example optimization
MLE(model, LBFGS())
```

You can pass extra keyword arguments, for example to set the maximum number of iterations.
These are all forwarded to `Optimization.solve`; please see [the Optimization docs](https://docs.sciml.ai/Optimization/stable/API/solve/) for more information.

```@example optimization
MLE(model, LBFGS(); maxiters=10)
```

!!! note "Transformations"

    Wren's optimisation routines are always performed in unconstrained coordinates.
    Internally this is accomplished by [vectorisation of the model](@ref vectorisation), which allows us to plug into the interface that Optimization.jl expects.

    However, the objective function that we maximise is always the log-density in the original space: the Jacobian term is not included.
    (Conceptually, this is because we are trying to find the parameters that maximise the original-space density, not the unconstrained-space density.)

## `optimize` in general

Both `MAP` and `MLE` are actually convenience wrappers around [`optimize`](@ref), which specify different objective functions to maximise.
Specifically, `MAP` selects `lp_function=logjoint`, and `MLE` selects `lp_function=loglikelihood`.

If you want to have a custom objective function, you can call `optimize` directly with the `lp_function` keyword argument.
Briefly speaking, the function you provide should map an `AccumulatorTuple` to a scalar log-density value.
You will most likely want to compose the `logprior` and `loglikelihood` primitives to construct your own objective function.
Here's a contrived example that demonstrates this:

```@example optimization
optimize(model, LBFGS(); lp_function=(accs -> logprior(accs) + 2 * loglikelihood(accs)))
```

## Initialisation

By default, optimisation starts from a draw from the model's priors.
You can choose an initialisation strategy for the optimisation using the `init` keyword argument.

`init` can either be an [`AbstractInitialization`](@ref) such as [`FindFiniteDensity`](@ref) (i.e., try until we get a starting point that has finite log-density), or something that can be converted to it.
Some examples are:

```@example optimization
# Initialise from [-2, 2] in unconstrained space
initial = FindFiniteDensity(FromUnconstrainedInterval(-2, 2))

# Draw from a seeded prior
using Random: Xoshiro
initial = FindFiniteDensity(FromPrior(Xoshiro(468)))

# Specific values, as a VarNamedTuple in model space.
initial = VarNamedTuple(rate=1.5)

# then:
MAP(model, LBFGS(); init=initial)
```
