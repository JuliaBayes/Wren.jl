"""
    rwm([rng], model, niters, nchains=1; step_size=nothing, warmup=0,
        progress=true, track=AllQuantities(), max_init_attempts=1000, adtype=nothing) -> VNChain

Gaussian random-walk Metropolis for fixed, continuous latents. The proposal standard
deviation `step_size` defaults to `2.38 / sqrt(d)` in unconstrained space.
Use `adtype=AutoReactant()` to run the sampler on the device.
"""
rwm(args...; kwargs...) = _random_walk(Val(:rwm), args...; kwargs...)

_random_walk_factor(::Val{:rwm}, step, d) = Diagonal(fill(step, d))
_walk_adaptation(::Val{:rwm}, x, scale, variance) = nothing
