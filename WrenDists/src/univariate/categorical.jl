struct Categorical{V<:AbstractVector{<:Number}} <: DiscreteUnivariateDist
    p::V
end

Base.eltype(::Type{Categorical{V}}) where {V} = float(eltype(V))

@inline function massat(d::Categorical, x::Number)
    T = masstype(d, x)
    m = zero(T)
    for (i, pᵢ) in enumerate(d.p)
        m += select(x == i, () -> convert(T, pᵢ), () -> zero(T))
    end
    return m
end

@inline function DensityInterface.logdensityof(d::Categorical, x::Number)
    return logt(massat(d, x))
end

# Some tools cannot stop a loop based on `q`, so count every partial sum below it.
@inline function _categorical_quantile(d::Categorical, q::Number)
    T = masstype(d, q)
    total = zero(T)
    i = one(T)
    for pᵢ in d.p
        total += convert(T, pᵢ)
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    return min(i, convert(T, length(d.p)))
end

# A sample changes in steps as the probabilities change, so its derivative is zero
# almost everywhere.
@inline function Base.rand(rng::AbstractRNG, d::Categorical)
    return _categorical_quantile(d, rand(rng, noisetype(d)))
end
