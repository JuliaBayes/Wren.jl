using LinearAlgebra: lmul!, lowrankdowndate!, lowrankupdate!, norm

struct _RAM{T<:Real}
    target_acceptance::T
end

"""
    ram([rng], model, niters, nchains=1; warmup=1000, target_acceptance=0.234,
        kwargs...) -> VNChain

Robust Adaptive Metropolis ([Vihola](https://arxiv.org/abs/1011.4381)). Adapt the proposal
covariance during warmup to target an acceptance rate of `target_acceptance`.
Other options match [`rwm`](@ref).
"""
function ram(args...; warmup::Integer=1000, target_acceptance::Real=0.234, kwargs...)
    0 < target_acceptance < 1 || throw(ArgumentError("target_acceptance must be in (0, 1)"))
    return _random_walk(_RAM(target_acceptance), args...; warmup, kwargs...)
end

_walk_adaptation(ram::_RAM, x, scale, variance) = ram

function _adapt_walk!(ram::_RAM, factor, x, noise, log_alpha, i, last)
    T = eltype(noise)
    gain = min(one(T), length(x) * T(i)^(-T(2) / T(3)))
    change = gain * (T(exp(log_alpha)) - T(ram.target_acceptance))
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
