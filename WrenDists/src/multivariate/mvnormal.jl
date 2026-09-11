const MvNormalFactor = Union{AbstractMatrix{<:Number},UniformScaling{<:Number}}

struct MvNormal{V<:AbstractVector{<:Number},F<:MvNormalFactor} <: MultivariateDist
    μ::V
    L::F
end

Base.size(d::MvNormal) = (length(d.μ),)

const DiagMvNormal = MvNormal{<:AbstractVector{<:Number},<:Diagonal}
const IsoMvNormal = MvNormal{<:AbstractVector{<:Number},<:UniformScaling}

@inline function shapesmatch(d::MvNormal, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1) & (size(d.L, 1) == n) & (size(d.L, 2) == n)
end

@inline function shapesmatch(d::DiagMvNormal, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1) & (length(d.L.diag) == n)
end

@inline function shapesmatch(d::IsoMvNormal, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1)
end

function logdetfactor(d::MvNormal, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for i in 1:length(d.μ)
        acc += logt(convert(R, d.L[i, i]))
    end
    return acc
end

function logdetfactor(d::DiagMvNormal, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for σ in d.L.diag
        acc += logt(convert(R, σ))
    end
    return acc
end

function logdetfactor(d::IsoMvNormal, ::Type{T}) where {T}
    return length(d.μ) * logt(convert(float(T), d.L.λ))
end

whiten(d::MvNormal, x::AbstractVector{<:Number}) = LowerTriangular(d.L) \ (x .- d.μ)

whiten(d::DiagMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.diag
whiten(d::IsoMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.λ

unwhiten(d::MvNormal, z::AbstractVector{<:Number}) = d.μ .+ LowerTriangular(d.L) * z

unwhiten(d::DiagMvNormal, z::AbstractVector{<:Number}) = d.μ .+ d.L.diag .* z
unwhiten(d::IsoMvNormal, z::AbstractVector{<:Number}) = d.μ .+ d.L.λ .* z

@inline function _densitytype(d::MvNormal, x::AbstractVector)
    return float(promote_type(eltype(d.μ), eltype(d.L), eltype(x)))
end

function DensityInterface.logdensityof(d::MvNormal, x::AbstractVector{<:Number})
    shapesmatch(d, x) || return convert(_densitytype(d, x), NaN)
    q = sum(abs2, whiten(d, x))
    R = float(typeof(q))
    n = length(d.μ)
    return -(q + n * convert(R, log2π)) / 2 - logdetfactor(d, R)
end

# Draw plain noise `z`, then return `μ + Lz`.
function Base.rand(rng::AbstractRNG, d::MvNormal)
    return unwhiten(d, randn(rng, noisetype(d), length(d.μ)))
end

# --- Batched MvNormal ---

# Whiten all draws together. Reduce the diagonal to avoid scalar indexing in logdet.
_whiten_batch(L::UniformScaling, X::AbstractMatrix) = X ./ L.λ
_whiten_batch(::UniformScaling{Bool}, X::AbstractMatrix) = X
_whiten_batch(L::Diagonal, X::AbstractMatrix) = X ./ L.diag
_whiten_batch(L::AbstractMatrix, X::AbstractMatrix) = LowerTriangular(L) \ X
_logdet_factor_batch(L::UniformScaling, n, ::Type{T}) where {T} = n * logt(T(L.λ))
_logdet_factor_batch(L::Diagonal, n, ::Type{T}) where {T} = sum(x -> logt(T(x)), L.diag)
_logdet_factor_batch(L::AbstractMatrix, n, ::Type{T}) where {T} =
    sum(x -> logt(T(x)), diag(L))
_centered_batch(X::AbstractMatrix, μ::AbstractVector) = X .- μ
function _iid_logdensityof(dist::MvNormal, x::AbstractArray)
    n = length(dist.μ)
    X = reshape(x, n, :)
    z2 = sum(abs2, _whiten_batch(dist.L, _centered_batch(X, dist.μ)))
    T = float(typeof(z2))
    lognorm = -(n * T(log2π)) / 2 - _logdet_factor_batch(dist.L, n, T)
    return size(X, 2) * lognorm - z2 / 2
end
