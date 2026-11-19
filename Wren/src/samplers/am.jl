using LinearAlgebra: Symmetric, cholesky, diagind, mul!

"""
    am([rng], model, niters, nchains=1; warmup=1000, kwargs...) -> VNChain

Adaptive Metropolis ([Haario et al.](https://doi.org/10.2307/3318737)). Tune the proposal
using the empirical covariance of all warmup states. Other options match [`rwm`](@ref).
"""
am(args...; warmup::Integer=1000, kwargs...) =
    _random_walk(Val(:am), args...; warmup, kwargs...)

struct _AMCovariance{T}
    mean::Vector{T}
    scatter::Matrix{T}
    scale_squared::T
    ridge::T
end

function _walk_adaptation(::Val{:am}, x, scale, variance)
    T, d = eltype(x), length(x)
    ridge = max(T(1e-6), eps(T)) * variance
    return _AMCovariance(copy(x), zeros(T, d, d), abs2(scale), ridge)
end

function _adapt_walk!(moments::_AMCovariance, factor, x, noise, log_alpha, i, last)
    # Welford's update includes the initial state and repeats after rejection.
    T = eltype(x)
    delta = x - moments.mean
    moments.mean .+= delta ./ T(i + 1)
    mul!(moments.scatter, delta, transpose(delta), T(i) / T(i + 1), one(T))
    if i >= 25 && (i % 25 == 0 || last)
        covariance = (moments.scale_squared / T(i)) .* moments.scatter
        covariance[diagind(covariance)] .+= moments.ridge
        return cholesky(Symmetric(covariance, :L))
    end
    return factor
end
