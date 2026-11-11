# Pathfinder

[Pathfinder.jl](https://mlcolab.github.io/Pathfinder.jl/stable/) fits a Gaussian approximation
to the posterior in unconstrained coordinates. It provides approximate posterior draws and
can help initialise MCMC. Load Pathfinder to enable its Wren model methods:

```@example pathfinder; continued=true
using Wren
using Wren.Dists: Gamma, Poisson
using Random: Xoshiro
using Pathfinder

@wren function poisson_rate(count)
    rate ~ Gamma(2.0, 1.0)
    @observe count ~ Poisson(rate)
end

model = poisson_rate(3)
fit = multipathfinder(model, 100; rng=Xoshiro(42))
```

[`Pathfinder.multipathfinder`](@ref) accepts a model, a draw count and a keyword RNG.
It returns Pathfinder's native result and approximation diagnostics.
`draws_transformed` is a model-space `FlexiChains.VNChain`, ready for plotting, summaries
and prediction:

```@example pathfinder
chain = fit.draws_transformed
mean(chain[@vn(rate)])
```

`fit.draws` contains unconstrained parameter vectors, one per column.
Both `fit.fit_distribution` and `fit.fit_distribution_transformed` remain in unconstrained coordinates.
The density includes the change-of-variables Jacobian,
as for NUTS, rather than the original-space MAP objective.

## Single and multiple paths

[`Pathfinder.pathfinder`](@ref) fits one Gaussian approximation and returns a native
`PathfinderResult`. Its draw count is a keyword:

```@example pathfinder
single = pathfinder(model; rng=Xoshiro(43), ndraws=100)
single.draws_transformed
```

For `multipathfinder`, four Gaussian fits form the mixture used for Pareto-smoothed
importance resampling by default. Set `nruns` to change the number of fits:

```@example pathfinder
fit = multipathfinder(model, 100; rng=Xoshiro(44), nruns=8)
fit.psis_result.pareto_shape
```

The fits are mixture components, not independent MCMC chains. All draws share one chain
index, regardless of `nruns`. Check the ELBO estimates in `fit.pathfinder_results`
and the Pareto shape diagnostic when assessing the approximation. Each path also has
model-space draws in `draws_transformed`. These are approximate
draws, not converged MCMC samples.

## Options

Both methods accept `adtype`, as for [`nuts`](@ref), and forward Pathfinder options
such as `history_length` and solver tolerances. `reduce_copies` has the same meaning as for
[`VecModel`](@ref).

`init` sets the optimiser's starting point in model space, as a latent source such as
`FromValues(VarNamedTuple(rate=2.0))`. By default, it is drawn from the prior. For
`multipathfinder`, `init` is a collection of latent sources whose length determines the
number of fits instead of `nruns`. If an optimisation run fails, Pathfinder retries from a
new starting point drawn from the prior. Pathfinder's own `init_sampler` and `init_scale`
options, which work in unconstrained coordinates, are not supported.

Both methods require `ntasks=1`; `multipathfinder` also requires `ntasks_per_run=1`.
Wren's model arguments and AD preparation may hold mutable state, so both methods run
sequentially. It copies the model arguments before preparing the density.
