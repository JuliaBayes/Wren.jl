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

function _delayed_log_acceptance(current, first, second, log_alpha1, noise)
    # The reverse first-stage move would always accept if first >= second.
    first < second || return -Inf
    # The second Gaussian proposal is symmetric. Only the first Gaussian and
    # first-stage rejection probabilities enter the correction.
    log_proposal_ratio = dot(noise.first, noise.retry) - dot(noise.retry, noise.retry) / 2
    log_reverse_reject = log(-expm1(first - second))
    log_forward_reject = log(-expm1(log_alpha1))
    log_ratio =
        (second - current) + log_proposal_ratio + log_reverse_reject - log_forward_reject
    return min(0, log_ratio)
end

_random_walk_stats(::_DRAM, log_density, stage) =
    (; accepted=stage != 0, log_density, accepted_stage=stage)
