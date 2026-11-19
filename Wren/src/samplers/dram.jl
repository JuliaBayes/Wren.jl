using LinearAlgebra: dot

struct _DRAM{T<:Real}
    second_scale::T
end

"""
    dram(
        [rng::Random.AbstractRNG=default_rng()],
        model::WrenCore.Model,
        niters::Integer,
        nchains::Integer=1;
        step_size::Union{Nothing,Real}=nothing,
        warmup::Integer=1000,
        second_scale::Real=0.5,
        progress::Bool=true,
        track::TrackedQuantitySpec=AllQuantities(),
        max_init_attempts::Integer=1000,
        adtype::Union{Nothing,AutoReactant}=nothing,
    ) -> FlexiChains.VNChain

Delayed Rejection Adaptive Metropolis for fixed, continuous latents
([Haario et al.](https://doi.org/10.1007/s11222-006-9438-0)).

- `second_scale` scales the second proposal's standard deviation.
- `accepted_stage` records the accepted stage, with 0 for rejection.
- `step_size` sets the initial proposal standard deviation in unconstrained space.
  The default is `2.38 / sqrt(d)` in `d` dimensions.
- `max_init_attempts` caps the prior draws used to find a finite starting log density.
- `adtype=AutoReactant()` runs transitions on the device.
"""
function dram(args...; warmup::Integer=1000, second_scale::Real=0.5, kwargs...)
    0 < second_scale < 1 || throw(ArgumentError("second_scale must be in (0, 1)"))
    return _random_walk(_DRAM(second_scale), args...; warmup, kwargs...)
end

_walk_adaptation(::_DRAM, x, scale, variance) =
    _walk_adaptation(Val(:am), x, scale, variance)

_device_walk_adaptation(backend, ::_DRAM, x, scale, variance) =
    _device_walk_adaptation(backend, Val(:am), x, scale, variance)

function _delayed_log_acceptance(current, first, second, log_alpha1, noise)
    # The second Gaussian proposal is symmetric. Only the first Gaussian and
    # first-stage rejection probabilities enter the correction.
    log_proposal_ratio = dot(noise.first, noise.retry) - dot(noise.retry, noise.retry) / 2
    log_reverse_reject = log(-expm1(min(0, first - second)))
    log_forward_reject = log(-expm1(log_alpha1))
    log_ratio =
        (second - current) + log_proposal_ratio + log_reverse_reject - log_forward_reject
    return ifelse(first < second, min(0, log_ratio), oftype(log_ratio, -Inf))
end

_random_walk_stats(::_DRAM, log_density, stage) =
    (; accepted=stage != 0, log_density, accepted_stage=stage)

function _device_dram_move(backend, algorithm, log_density, rng, current, first, accepted)
    retry = Base.Fix1(_device_dram_retry, eltype(current.x)(algorithm.second_scale))
    return backend._device_if(
        _device_dram_first,
        retry,
        accepted,
        rng,
        log_density,
        current,
        first,
    )
end

function _device_dram_first(rng, log_density, current, first)
    valid = first.density < Inf
    return first.proposal, first.density, ifelse(valid, 1, 0), !valid
end

function _device_dram_retry(scale, rng, log_density, current, first)
    (; x, density, factor) = current
    retry = scale .* Random.randn(rng, eltype(x), length(x))
    second = _device_walk_proposal(x, factor, retry)
    second_density = _device_walk_score(log_density, second)
    log_alpha2 = _delayed_log_acceptance(
        density,
        first.density,
        second_density,
        first.log_alpha,
        (; first=first.noise, retry),
    )
    accepted = log(Random.rand(rng)) < log_alpha2
    return second, second_density, ifelse(accepted, 2, 0), !(second_density < Inf)
end
