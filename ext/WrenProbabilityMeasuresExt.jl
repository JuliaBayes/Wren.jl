module WrenProbabilityMeasuresExt

using Wren: Wren
using Plaice: Plaice
using Random: Random
using LinearAlgebra: Diagonal, LowerTriangular, UniformScaling, diag
using DensityInterface: DensityInterface, logdensityof
using ProbabilityMeasures: ProbabilityMeasures as PM

"""
    Repeated{D,N}

A single measure `dist` repeated over `N` extra dimensions `dims`. Used as the component
collection of an [`Independent`](@ref) built by `Wren.iid`.
"""
struct Repeated{D<:PM.AbstractProbabilityMeasure,N}
    dist::D
    dims::Dims{N}
end
Base.size(r::Repeated) = (size(r.dist)..., r.dims...)
Base.:(==)(a::Repeated, b::Repeated) = (a.dist == b.dist) & (a.dims == b.dims)

"""
    Independent{C}

The product measure of a collection `components` of measures. Construct it with
`Wren.independent(dists::AbstractArray)` or `Wren.iid(dist, dims...)`.
"""
struct Independent{C}
    components::C
end

const IID{D,N} = Independent{Repeated{D,N}}
const IIDUnivariate = Independent{<:Repeated{<:PM.UnivariateMeasure}}
const IIDMultivariate = Independent{<:Repeated{<:PM.MultivariateMeasure}}
const IndependentArray = Independent{<:AbstractArray{<:PM.UnivariateMeasure}}

Wren.iid(dist::PM.AbstractProbabilityMeasure, dims::Dims) =
    Independent(Repeated(dist, dims))
Wren.independent(dists::AbstractArray{<:PM.UnivariateMeasure}) = Independent(dists)
function Wren.independent(dists::AbstractArray{<:PM.MultivariateMeasure})
    throw(
        ArgumentError(
            "independent products of multivariate ProbabilityMeasures are not supported; " *
            "use `iid` for identically distributed draws or a single multivariate measure.",
        ),
    )
end

_dist(d::IID) = d.components.dist
_dims(d::IID) = d.components.dims

Base.size(d::Independent) = size(d.components)
Base.length(d::Independent) = prod(size(d))
Base.broadcastable(d::Independent) = Ref(d)
Base.:(==)(a::Independent, b::Independent) = a.components == b.components
Base.show(io::IO, d::IID) = print(io, "iid(", _dist(d), ", ", _dims(d), ")")
Base.show(io::IO, d::Independent) = print(io, "independent(", d.components, ")")

Base.rand(rng::Random.AbstractRNG, d::IIDUnivariate) = rand(rng, _dist(d), _dims(d))
# Stack vector draws along the trailing dimensions.
function Base.rand(rng::Random.AbstractRNG, d::IIDMultivariate)
    draws = rand(rng, _dist(d), _dims(d))
    isempty(draws) && return Array{eltype(eltype(_dist(d)))}(undef, size(d))
    return stack(draws)
end
Base.rand(rng::Random.AbstractRNG, d::IndependentArray) =
    map(di -> rand(rng, di), d.components)
Base.rand(d::Independent) = rand(Random.default_rng(), d)

DensityInterface.DensityKind(::Independent) = DensityInterface.HasDensity()
function _check_size(d::Independent, x::AbstractArray)
    size(x) == size(d) ||
        throw(DimensionMismatch("expected size $(size(d)), got $(size(x))"))
    return nothing
end
function DensityInterface.logdensityof(d::IID, x::AbstractArray)
    _check_size(d, x)
    return _iid_logdensityof(_dist(d), x)
end
function DensityInterface.logdensityof(d::IndependentArray, x::AbstractArray)
    _check_size(d, x)
    return sum(logdensityof.(d.components, x))
end

"""
    _iid_logdensityof(dist, x::AbstractArray)

Sum of `logdensityof(dist, xᵢ)` over every draw `xᵢ` stored in `x`. Add methods to this
function to provide closed-form expressions for specific measures.
"""
function _iid_logdensityof(dist::PM.UnivariateMeasure, x::AbstractArray)
    return sum(Base.Fix1(logdensityof, dist), x; init=_zero_logdensity(dist, x))
end

# Follows ProbabilityMeasures' result-type rule so empty arrays sum to a typed zero.
_draw_eltype(dist::PM.UnivariateMeasure) = eltype(dist)
_draw_eltype(dist::PM.MultivariateMeasure) = eltype(eltype(dist))
function _zero_logdensity(dist::PM.AbstractProbabilityMeasure, x::AbstractArray)
    return zero(float(promote_type(eltype(x), _draw_eltype(dist))))
end

# Each slice is one draw. This fallback is CPU-only.
function _iid_logdensityof(
    dist::PM.MultivariateMeasure,
    x::AbstractArray{<:Any,M},
) where {M}
    slices = eachslice(x; dims=ntuple(i -> i + 1, Val(M - 1)))
    return sum(Base.Fix1(logdensityof, dist), slices; init=_zero_logdensity(dist, x))
end

# Compute the shared normalising constant once.
function _iid_logdensityof(dist::PM.Normal, x::AbstractArray)
    μ = dist.μ
    _, _, σ = promote(zero(eltype(x)), μ, dist.σ)
    inv_σ = inv(float(σ))
    z2 = sum(xi -> abs2((xi - μ) * inv_σ), x; init=zero(inv_σ))
    lognorm = logdensityof(dist, oftype(z2, μ))
    return length(x) * lognorm - z2 / 2
end

# Whiten all draws together. Reduce the diagonal to avoid scalar indexing in PM.logdet.
_whiten(L::UniformScaling, X::AbstractMatrix) = X ./ L.λ
_whiten(::UniformScaling{Bool}, X::AbstractMatrix) = X
_whiten(L::Diagonal, X::AbstractMatrix) = X ./ L.diag
_whiten(L::AbstractMatrix, X::AbstractMatrix) = LowerTriangular(L) \ X
_logdet_factor(L::UniformScaling, n, ::Type{T}) where {T} = n * PM.logt(T(L.λ))
_logdet_factor(L::Diagonal, n, ::Type{T}) where {T} = sum(x -> PM.logt(T(x)), L.diag)
_logdet_factor(L::AbstractMatrix, n, ::Type{T}) where {T} = sum(x -> PM.logt(T(x)), diag(L))
_centered(X::AbstractMatrix, μ::AbstractVector) = X .- μ
function _iid_logdensityof(dist::PM.MvNormal, x::AbstractArray)
    n = length(dist.μ)
    X = reshape(x, n, :)
    z2 = sum(abs2, _whiten(dist.L, _centered(X, dist.μ)))
    T = float(typeof(z2))
    lognorm = -(n * T(PM.log2π)) / 2 - _logdet_factor(dist.L, n, T)
    return size(X, 2) * lognorm - z2 / 2
end

# Apply one bijector or an array of bijectors, flattening on link and reshaping on inverse.
struct ElementwiseLink{B,N} <: Plaice.AbstractBijector
    bijector::B
    dims::Dims{N}
end
struct ElementwiseInvLink{B,N} <: Plaice.AbstractBijector
    bijector::B
    dims::Dims{N}
end

_map_inverse(b) = Plaice.inverse(b)
_map_inverse(b::AbstractArray) = Plaice.inverse.(b)
Plaice.inverse(t::ElementwiseLink) = ElementwiseInvLink(_map_inverse(t.bijector), t.dims)
Plaice.inverse(t::ElementwiseInvLink) = ElementwiseLink(_map_inverse(t.bijector), t.dims)

for T in (:ElementwiseLink, :ElementwiseInvLink)
    @eval begin
        Base.:(==)(a::$T, b::$T) = (a.bijector == b.bijector) & (a.dims == b.dims)
    end
end

_bcast(b) = Ref(b)
_bcast(b::AbstractArray) = b

# Compute Jacobians separately to avoid broadcasting tuples.
_ladj(b, x) = Plaice.logabsdet_jacobian(b, x)
_ladj(::Plaice.TypedIdentity, x) = zero(x)
_ladj(::Plaice.Exp, y) = y
_ladj(b::Plaice.Log, x) = -log(b.sign * (x - b.bound))
_apply(b, x) = b(x)

const IdentityLike = Union{Plaice.TypedIdentity,AbstractArray{Plaice.TypedIdentity}}

function Plaice.with_logabsdet_jacobian(t::ElementwiseLink, x::AbstractArray)
    b = _bcast(t.bijector)
    return vec(_apply.(b, x)), sum(_ladj.(b, x))
end
(t::ElementwiseLink)(x::AbstractArray) = vec(_apply.(_bcast(t.bijector), x))

function Plaice.with_logabsdet_jacobian(t::ElementwiseInvLink, y::AbstractVector)
    b = _bcast(t.bijector)
    yr = reshape(y, t.dims)
    return _apply.(b, yr), sum(_ladj.(b, yr))
end
(t::ElementwiseInvLink)(y::AbstractVector) = _apply.(_bcast(t.bijector), reshape(y, t.dims))

# Identity transforms only change the shape.
function Plaice.with_logabsdet_jacobian(
    ::ElementwiseLink{<:IdentityLike},
    x::AbstractArray{T},
) where {T<:Number}
    return vec(x), zero(T)
end
(::ElementwiseLink{<:IdentityLike})(x::AbstractArray) = vec(x)
function Plaice.with_logabsdet_jacobian(
    t::ElementwiseInvLink{<:IdentityLike},
    y::AbstractVector{T},
) where {T<:Number}
    return reshape(y, t.dims), zero(T)
end
(t::ElementwiseInvLink{<:IdentityLike})(y::AbstractVector) = reshape(y, t.dims)

# For Exp, the log-Jacobian is sum(y).
function Plaice.with_logabsdet_jacobian(
    t::ElementwiseInvLink{<:Plaice.Exp},
    y::AbstractVector,
)
    yr = reshape(y, t.dims)
    return _apply.(Ref(t.bijector), yr), sum(y)
end

Plaice.from_vec(d::Independent) = Plaice.Reshape(size(d))
Plaice.to_vec(d::Independent) = Plaice.Vec(size(d))
Plaice.vec_length(d::Independent) = length(d)
Plaice.unconstrained_vec_length(d::Independent) = length(d)

function Plaice.from_unconstrained_vec(d::IIDUnivariate)
    return ElementwiseInvLink(
        Plaice.inverse(Plaice.scalar_to_scalar_bijector(_dist(d))),
        _dims(d),
    )
end
function Plaice.to_unconstrained_vec(d::IIDUnivariate)
    return ElementwiseLink(Plaice.scalar_to_scalar_bijector(_dist(d)), _dims(d))
end

function _check_identity_link(d::IIDMultivariate)
    Plaice.from_unconstrained_vec(_dist(d)) isa Plaice.TypedIdentity || throw(
        ArgumentError(
            "iid over a multivariate measure is only supported when the measure is " *
            "already unconstrained; got $(_dist(d)).",
        ),
    )
    return nothing
end
function Plaice.from_unconstrained_vec(d::IIDMultivariate)
    _check_identity_link(d)
    return Plaice.Reshape(size(d))
end
function Plaice.to_unconstrained_vec(d::IIDMultivariate)
    _check_identity_link(d)
    return Plaice.Vec(size(d))
end

function Plaice.from_unconstrained_vec(d::IndependentArray)
    return ElementwiseInvLink(
        Plaice.inverse.(Plaice.scalar_to_scalar_bijector.(d.components)),
        size(d),
    )
end
function Plaice.to_unconstrained_vec(d::IndependentArray)
    return ElementwiseLink(Plaice.scalar_to_scalar_bijector.(d.components), size(d))
end

Wren.batched(iters, dist::PM.AbstractProbabilityMeasure) = Wren.iid(dist, size(iters))
Wren._product(dists::AbstractArray{<:PM.AbstractProbabilityMeasure}) =
    Wren.independent(dists)

struct DiagNormal{M<:AbstractArray,S<:AbstractArray}
    mu::M
    sigma::S
end
Plaice.from_unconstrained_vec(d::DiagNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::DiagNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::DiagNormal) =
    randn(rng, size(d.mu)) .* d.sigma .+ d.mu
@inline function Wren.logdensityof(d::DiagNormal, x)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x, d.mu, d.sigma)
        z = (x[i] - d.mu[i]) / d.sigma[i]
        s += z * z + 2 * log(d.sigma[i])
    end
    return -0.5 * s - 0.5 * length(d.mu) * log(2π)
end
function Wren.batched(
    iters,
    ::Val{PM.Normal},
    indexed_mu::Wren.Indexed,
    indexed_sigma::Wren.Indexed,
)
    # e.g. batched(1:10, Val(PM.Normal), Indexed(x[1:10]), Indexed(y[1:10]))
    return DiagNormal(indexed_mu.data, indexed_sigma.data)
end

struct IsoNormal{M<:AbstractArray}
    mu::M
end
Plaice.from_unconstrained_vec(d::IsoNormal) = Plaice.Reshape(size(d.mu))
Plaice.to_unconstrained_vec(d::IsoNormal) = Plaice.Vec(size(d.mu))
Base.rand(rng::Random.AbstractRNG, d::IsoNormal) = randn(rng, size(d.mu)) .+ d.mu
function Wren.logdensityof(d::IsoNormal, x)
    dist = sum(abs2, x .- d.mu)
    return -0.5 * dist - 0.5 * length(d.mu) * log(2π)
end
function Wren.batched(iters, ::Val{PM.Normal}, indexed_mu::Wren.Indexed)
    # e.g. batched(1:10, Val(PM.Normal), Indexed(x[1:10]))
    return IsoNormal(indexed_mu.data)
end

end # module WrenProbabilityMeasuresExt
