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

_am_ridge(T, variance) = max(T(1e-6), eps(T)) * variance
_am_refresh(i, last) = (i >= 25) & ((i % 25 == 0) | last)

function _walk_adaptation(::Val{:am}, x, scale, variance)
    T, d = eltype(x), length(x)
    return _AMCovariance(copy(x), zeros(T, d, d), abs2(scale), _am_ridge(T, variance))
end

function _adapt_walk!(moments::_AMCovariance, factor, x, noise, log_alpha, i, last)
    # Welford's update includes the initial state and repeats after rejection.
    T = eltype(x)
    delta = x - moments.mean
    moments.mean .+= delta ./ T(i + 1)
    mul!(moments.scatter, delta, transpose(delta), T(i) / T(i + 1), one(T))
    if _am_refresh(i, last)
        covariance = (moments.scale_squared / T(i)) .* moments.scatter
        covariance[diagind(covariance)] .+= moments.ridge
        return cholesky(Symmetric(covariance, :L))
    end
    return factor
end

function _device_walk_adaptation(backend, ::Val{:am}, x, scale, variance)
    T, d = eltype(x), length(x)
    mean = backend._device_array(x)
    scatter = backend._device_array(zeros(T, d, d))
    ridge = _am_ridge(T, variance)
    ridge_matrix = backend._device_array(Matrix(Diagonal(fill(ridge, d))))
    return (; mean, scatter, scale_squared=abs2(scale), ridge_matrix)
end

function _device_adapt_walk(backend, moments::NamedTuple, factor, x, move, i, last)
    (; mean, scatter, scale_squared, ridge_matrix) = moments
    T = eltype(x)
    delta = x - mean
    mean += delta ./ T(i + 1)
    scatter += (T(i) / T(i + 1)) .* (delta * transpose(delta))
    refactor =
        (factor, scatter, ridge, i) ->
            backend._device_cholesky((scale_squared / T(i)) .* scatter + ridge)
    keep = (factor, scatter, ridge, i) -> factor
    factor = backend._device_if(
        refactor,
        keep,
        _am_refresh(i, last),
        factor,
        scatter,
        ridge_matrix,
        i,
    )
    return factor, (; mean, scatter, scale_squared, ridge_matrix)
end
