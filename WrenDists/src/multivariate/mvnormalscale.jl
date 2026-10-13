const MvNormalScaleFactor = Union{AbstractMatrix{<:Number},UniformScaling{<:Number}}

"""
    MvNormalScale(μ, L)

Multivariate normal distribution with mean vector `μ` and scale factor `L`, where `L` can be
a lower-triangular matrix, a `Diagonal`, or a `UniformScaling`.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)

!!! warning

    This differs from Distributions.jl which uses the covariance matrix `Σ` instead of the
    scale factor `L`. The relationship is `Σ = L * L'`. The intention of this is to avoid
    the need to compute the Cholesky factor of `Σ` inside the model when it is already
    known, which is often the case with standard deviations that are constant.

    In particular, if the covariance matrix is diagonal, then `L` is simply the vector of
    standard deviations (rather than variances, as in Distributions.jl).
"""
struct MvNormalScale{V<:AbstractVector{<:Number},F<:MvNormalScaleFactor} <: MultivariateDist
    μ::V
    L::F
end

Base.size(d::MvNormalScale) = (length(d.μ),)

const DiagMvNormalScale = MvNormalScale{<:AbstractVector{<:Number},<:Diagonal}
const IsoMvNormalScale = MvNormalScale{<:AbstractVector{<:Number},<:UniformScaling}

@inline function shapesmatch(d::MvNormalScale, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1) & (size(d.L, 1) == n) & (size(d.L, 2) == n)
end

@inline function shapesmatch(d::DiagMvNormalScale, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1) & (length(d.L.diag) == n)
end

@inline function shapesmatch(d::IsoMvNormalScale, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1)
end

function logdetfactor(d::MvNormalScale, ::Type{T}) where {T}
    R = float(T)
    return if within_compile()
        sum(x -> log(convert(R, x)), diag(d.L))
    else
        acc = zero(R)
        for i in 1:length(d.μ)
            acc += log(convert(R, d.L[i, i]))
        end
        acc
    end
end

function logdetfactor(d::DiagMvNormalScale, ::Type{T}) where {T}
    R = float(T)
    return if within_compile()
        sum(σ -> log(convert(R, σ)), d.L.diag)
    else
        acc = zero(R)
        for σ in d.L.diag
            acc += log(convert(R, σ))
        end
        acc
    end
end

function logdetfactor(d::IsoMvNormalScale, ::Type{T}) where {T}
    return length(d.μ) * log(convert(float(T), d.L.λ))
end

whiten(d::MvNormalScale, x::AbstractVector{<:Number}) = LowerTriangular(d.L) \ (x .- d.μ)

whiten(d::DiagMvNormalScale, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.diag
whiten(d::IsoMvNormalScale, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.λ

unwhiten(d::MvNormalScale, z::AbstractVector{<:Number}) = d.μ .+ LowerTriangular(d.L) * z

unwhiten(d::DiagMvNormalScale, z::AbstractVector{<:Number}) = d.μ .+ d.L.diag .* z
unwhiten(d::IsoMvNormalScale, z::AbstractVector{<:Number}) = d.μ .+ d.L.λ .* z

@inline function _densitytype(d::MvNormalScale, x::AbstractVector)
    return float(promote_type(eltype(d.μ), eltype(d.L), eltype(x)))
end

function DensityInterface.logdensityof(d::MvNormalScale, x::AbstractVector{<:Number})
    shapesmatch(d, x) || return convert(_densitytype(d, x), NaN)
    q = sum(abs2, whiten(d, x))
    R = float(typeof(q))
    n = length(d.μ)
    return -(q + n * convert(R, log2π)) / 2 - logdetfactor(d, R)
end

# Draw plain noise `z`, then return `μ + Lz`.
function Base.rand(rng::AbstractRNG, d::MvNormalScale)
    return unwhiten(d, randn(rng, noisetype(d), length(d.μ)))
end

# --- Batched MvNormalScale ---

# Whiten all draws together. Reduce the diagonal to avoid scalar indexing in logdet.
_whiten_batch(L::UniformScaling, X::AbstractMatrix) = X ./ L.λ
_whiten_batch(::UniformScaling{Bool}, X::AbstractMatrix) = X
_whiten_batch(L::Diagonal, X::AbstractMatrix) = X ./ L.diag
_whiten_batch(L::AbstractMatrix, X::AbstractMatrix) = LowerTriangular(L) \ X
_logdet_factor_batch(L::UniformScaling, n, ::Type{T}) where {T} = n * log(T(L.λ))
_logdet_factor_batch(L::Diagonal, n, ::Type{T}) where {T} = sum(x -> log(T(x)), L.diag)
_logdet_factor_batch(L::AbstractMatrix, n, ::Type{T}) where {T} =
    sum(x -> log(T(x)), diag(L))
_centered_batch(X::AbstractMatrix, μ::AbstractVector) = X .- μ
function _iid_logdensityof(dist::MvNormalScale, x::AbstractArray)
    n = length(dist.μ)
    X = reshape(x, n, :)
    z2 = sum(abs2, _whiten_batch(dist.L, _centered_batch(X, dist.μ)))
    T = float(typeof(z2))
    lognorm = -(n * T(log2π)) / 2 - _logdet_factor_batch(dist.L, n, T)
    return size(X, 2) * lognorm - z2 / 2
end
