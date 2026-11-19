using LinearAlgebra: dot

struct _DRAM{T<:Real}
    second_scale::T
end

"""
    dram([rng], model, niters, nchains=1; warmup=1000, second_scale=0.5,
         kwargs...) -> VNChain

Delayed Rejection Adaptive Metropolis ([Haario et al.](https://doi.org/10.1007/s11222-006-9438-0)).
After rejection, try a second Gaussian proposal with step size multiplied by `second_scale`.
The chain records the accepted stage as `accepted_stage` (0 for rejection).
Adaptation follows [`am`](@ref). Other options match [`rwm`](@ref).
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

function _device_walk_move(backend, algorithm::_DRAM, log_density, rng, x, density, factor)
    noise = Random.randn(rng, eltype(x), length(x))
    proposal = _device_walk_proposal(x, factor, noise)
    proposed_density = _device_walk_score(log_density, proposal)
    log_alpha = min(0, proposed_density - density)
    accepted = log(Random.rand(rng)) < log_alpha
    retry = Base.Fix1(_device_dram_retry, eltype(x)(algorithm.second_scale))
    current = (; x, density, factor)
    first = (; proposal, density=proposed_density, noise, log_alpha)
    selected, selected_density, stage, invalid = backend._device_if(
        _device_dram_first,
        retry,
        accepted,
        rng,
        log_density,
        current,
        first,
    )
    invalid = invalid | !(proposed_density < Inf)
    return selected, selected_density, stage, noise, log_alpha, invalid
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
