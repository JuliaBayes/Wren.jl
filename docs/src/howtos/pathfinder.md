# Pathfinder

[Pathfinder.jl](https://mlcolab.github.io/Pathfinder.jl/stable/) fits a Gaussian approximation
to the posterior in unconstrained coordinates. It provides approximate posterior draws and
can help initialise MCMC. Load Pathfinder to enable its Wren model methods:

```@example pathfinder; continued=true
using Wren
using Wren.Dists: Gamma, Poisson
using Random: Xoshiro
import Pathfinder

@wren function poisson_rate(count)
    rate ~ Gamma(2.0, 1.0)
    @observe count ~ Poisson(rate)
end

model = poisson_rate(3)
fit = pathfinder(Xoshiro(42), model, 100)
```

Like [`nuts`](@ref), [`pathfinder`](@ref) accepts an optional positional RNG, a model and a
draw count. It returns Pathfinder's native result and approximation diagnostics.
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

## Multiple paths

By default, four Gaussian fits form the mixture used for Pareto-smoothed importance
resampling. Set `nruns` to change the number of fits:

```@example pathfinder
fit = pathfinder(Xoshiro(43), model, 100; nruns=8)
fit.psis_result.pareto_shape
```

The fits are mixture components, not independent MCMC chains. All draws share one chain
index, regardless of `nruns`. Check the ELBO estimates in `fit.pathfinder_results`
and the Pareto shape diagnostic when assessing the approximation. These are approximate
draws, not converged MCMC samples.

## Options

The method accepts `adtype`, as for [`nuts`](@ref), and forwards Pathfinder options
such as `init`, `init_sampler`, `history_length`, and solver tolerances. `init` and
`init_sampler` use unconstrained coordinates and retain Pathfinder's defaults.
When `init` is supplied, its length determines the number of fits instead of `nruns`.
`latent_source=FromPrior(rng)` prepares Wren's parameter mapping; it does not set the
optimiser's starting points. `reduce_copies` has the same meaning as for [`VecModel`](@ref).

The method requires `ntasks=1` and `ntasks_per_run=1`.
Wren's model arguments and AD preparation may hold mutable state, so the method runs
sequentially. It copies the model arguments before preparing the density.
