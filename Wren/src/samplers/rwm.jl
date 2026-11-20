"""
    rwm(
        [rng::Random.AbstractRNG=default_rng()],
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        step_size::Union{Nothing,Real}=nothing,
        warmup::Integer=0,
        progress::Bool=true,
        track::TrackedQuantitySpec=AllQuantities(),
        max_init_attempts::Integer=1000,
        with_reactant::Bool=false,
    ) -> FlexiChains.VNChain

Gaussian random-walk Metropolis for fixed, continuous latents.

- `step_size` sets the proposal standard deviation in unconstrained space.
  The default is `2.38 / sqrt(d)` in `d` dimensions.
- `max_init_attempts` caps the prior draws used to find a finite starting log density.
- `with_reactant=true` runs transitions on the device.
"""
rwm(args...; kwargs...) = _random_walk(Val(:rwm), args...; kwargs...)

_random_walk_factor(::Val{:rwm}, step, d) = Diagonal(fill(step, d))
_walk_adaptation(::Val{:rwm}, x, scale, variance) = nothing
