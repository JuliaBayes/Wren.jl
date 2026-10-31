# [Point estimates](@id point-estimates)

Load Optimization.jl and a solver package to find a maximum a posteriori (MAP) estimate:

```julia
using Wren
using Wren.Dists: Gamma, Poisson
using OptimizationOptimJL: LBFGS
import Enzyme, Optimization

@wren function poisson_rate(count)
    rate ~ Gamma(2.0, 1.0)
    @observe count ~ Poisson(rate)
end

model = poisson_rate(3)
initial = FromValues(VarNamedTuple(rate=1.5))
fit = Wren.optimize(model, LBFGS(); latent_source=initial)
fit.params[@vn(rate)]  # approximately 2
fit.solution          # native solver result, including convergence status
```

Use `lp_function=loglikelihood` for maximum likelihood (MLE):

```julia
fit = Wren.optimize(model, LBFGS(); latent_source=initial, lp_function=loglikelihood)
fit.params[@vn(rate)]  # approximately 3
```

Both methods optimise within the parameter support declared by the model's priors.
Parameters use unconstrained coordinates internally, but the objective is the original-space
density without a change-of-variables Jacobian. Solver keywords are passed to
`Optimization.solve`; check `fit.solution.retcode` before using an estimate.

For direct control, [`logdensity`](@ref) constructs the same [`VecModel`](@ref) using the
LogDensityProblems.jl interface:

```julia
import LogDensityProblems as LDP

density = logdensity(model, AutoEnzyme(; mode=Reverse); latent_source=initial)
x = vectorise_latents(initial, density)
LDP.logdensity(density, x)
LDP.logdensity_and_gradient(density, x)
```

The default is the unnormalised posterior density in unconstrained coordinates, including
the Jacobian used by samplers. For a custom optimisation problem, pass `lp_function=logjoint`
for the original-space MAP objective or `lp_function=loglikelihood` for MLE. Omit the AD
backend when only density values are needed. This interface does not require Optimization.jl.
