"""
    optimize(model, solver; adtype=DEFAULT_AD, latent_source=FromPrior(),
             lp_function=logjoint, reduce_copies=true, kwargs...)

Find a maximum a posteriori (MAP) estimate with an Optimization.jl solver. Requires
Optimization.jl and the selected AD backend to be loaded, as for [`VecModel`](@ref).
Use `lp_function=loglikelihood` for maximum likelihood (MLE).
Both objectives use the latent support declared by the model's priors.

Returns `(params, solution)`: named model-space parameters and the native Optimization
solution, including its convergence status. The solver minimises the negative original-space
log-density in unconstrained coordinates. Initialisation and AD preparation use [`logdensity`](@ref).
Solver keywords are forwarded to `Optimization.solve`.
"""
function optimize end
