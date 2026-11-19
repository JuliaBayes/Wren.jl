using LinearAlgebra: lowrankdowndate!, lowrankupdate!, norm

struct _RAM{T<:Real}
    target_acceptance::T
end

"""
    ram(
        [rng::Random.AbstractRNG=default_rng()],
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        step_size::Union{Nothing,Real}=nothing,
        warmup::Integer=1000,
        target_acceptance::Real=0.234,
        progress::Bool=true,
        track::TrackedQuantitySpec=AllQuantities(),
        max_init_attempts::Integer=1000,
        adtype::Union{Nothing,AutoReactant}=nothing,
    ) -> FlexiChains.VNChain

Robust Adaptive Metropolis for fixed, continuous latents ([Vihola](https://arxiv.org/abs/1011.4381)).

- `target_acceptance` sets the warmup acceptance target.
- `step_size` sets the initial proposal standard deviation in unconstrained space.
  The default is `2.38 / sqrt(d)` in `d` dimensions.
- `max_init_attempts` caps the prior draws used to find a finite starting log density.
- `adtype=AutoReactant()` runs transitions on the device.
"""
function ram(args...; warmup::Integer=1000, target_acceptance::Real=0.234, kwargs...)
    0 < target_acceptance < 1 || throw(ArgumentError("target_acceptance must be in (0, 1)"))
    return _random_walk(_RAM(target_acceptance), args...; warmup, kwargs...)
end

_walk_adaptation(ram::_RAM, x, scale, variance) = ram

function _ram_change(ram, noise, log_alpha, i)
    T = eltype(noise)
    gain = min(one(T), length(noise) * T(i)^(-T(2) / T(3)))
    return gain * (T(exp(log_alpha)) - T(ram.target_acceptance))
end

function _adapt_walk!(ram::_RAM, factor, x, noise, log_alpha, i, last)
    change = _ram_change(ram, noise, log_alpha, i)
    noise_norm = norm(noise)
    if !iszero(change) && !iszero(noise_norm)
        noise .*= sqrt(abs(change)) / noise_norm
        lmul!(factor.L, noise)
        if change > 0
            lowrankupdate!(factor, noise)
        else
            lowrankdowndate!(factor, noise)
        end
    end
    return factor
end

_device_walk_adaptation(backend, ram::_RAM, x, scale, variance) = ram

function _device_adapt_walk(backend, ram::_RAM, factor, x, move, i, last)
    (; noise, log_alpha) = move
    change = _ram_change(ram, noise, log_alpha, i)
    noise_norm = sqrt(sum(abs2, noise))
    direction = noise ./ ifelse(iszero(noise_norm), one(noise_norm), noise_norm)
    # The square root of I + change * u*u' is I + coefficient * u*u'.
    coefficient = change / (sqrt(1 + change) + 1)
    factor += (coefficient .* (factor * direction)) * transpose(direction)
    return factor, ram
end
