struct Multinomial{N<:Integer,V<:AbstractVector{<:Number}} <: DiscreteMultivariateDist
    n::N
    p::V
end

function Base.eltype(::Type{Multinomial{N,V}}) where {N,V}
    return Vector{float(eltype(V))}
end

function insupport(d::Multinomial, x::AbstractVector{<:Number})
    length(x) == length(d.p) || return false
    remaining = d.n
    ok = true
    for xᵢ in x
        valid = isfinite(xᵢ) & isinteger(xᵢ) & (xᵢ >= zero(xᵢ)) & (xᵢ <= remaining)
        ok &= valid
        remaining -= select(valid, () -> xᵢ, () -> zero(xᵢ))
    end
    return ok & iszero(remaining)
end

function DensityInterface.logdensityof(d::Multinomial, x::AbstractVector{<:Number})
    T = float(promote_type(typeof(d.n), eltype(d.p), eltype(x)))
    length(x) == length(d.p) || return convert(T, -Inf)
    n = convert(T, d.n)
    logp = loggamma(max(n, zero(T)) + one(T))
    for i in eachindex(d.p)
        xᵢ, pᵢ = convert(T, x[i]), convert(T, d.p[i])
        logp -= loggamma(max(xᵢ, zero(T)) + one(T))
        logp += select(xᵢ == zero(T), () -> zero(T), () -> xᵢ * logt(pᵢ))
    end
    result = logp
    return select(insupport(d, x), () -> result, () -> convert(T, -Inf))
end

function Base.rand(rng::AbstractRNG, d::Multinomial)
    T = eltype(eltype(d))
    counts = zeros(T, length(d.p))
    categorical = Categorical(d.p)
    for _ in 1:(d.n)
        category = rand(rng, categorical)
        for i in eachindex(counts)
            counts[i] += select(category == i, () -> one(T), () -> zero(T))
        end
    end
    return counts
end
