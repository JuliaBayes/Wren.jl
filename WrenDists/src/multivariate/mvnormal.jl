const MvNormalFactor = Union{AbstractMatrix{<:Number},UniformScaling{<:Number}}

struct MvNormal{V<:AbstractVector{<:Number},F<:MvNormalFactor} <: WrenDist
    μ::V
    L::F
end

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

@inline function rowdot(L::AbstractMatrix, v::AbstractVector, i::Integer, k::Integer)
    acc = L[i, 1] * v[1]
    for j in 2:k
        acc = muladd(L[i, j], v[j], acc)
    end
    return acc
end

function whiten(d::MvNormal, x::AbstractVector{<:Number})
    μ, L = d.μ, d.L
    z = [(x[1] - μ[1]) / L[1, 1]]
    for i in 2:length(μ)
        z = vcat(z, (x[i] - μ[i] - rowdot(L, z, i, i - 1)) / L[i, i])
    end
    return z
end

whiten(d::DiagMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.diag
whiten(d::IsoMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.λ

function unwhiten(d::MvNormal, z::AbstractVector{<:Number})
    return map(i -> d.μ[i] + rowdot(d.L, z, i, i), 1:length(d.μ))
end

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
