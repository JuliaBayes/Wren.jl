struct Dirichlet{V<:AbstractVector{<:Number}} <: MultivariateDist
    α::V
end

Base.size(d::Dirichlet) = (length(d.α),)

Base.eltype(::Type{Dirichlet{V}}) where {V} = Vector{float(eltype(V))}

function checkparams(d::Dirichlet)
    isempty(d.α) && return false
    ok = true
    for αᵢ in d.α
        ok &= isfinite(αᵢ) & (αᵢ > zero(αᵢ))
    end
    return ok
end

function insupport(d::Dirichlet, x::AbstractVector{<:Number})
    length(x) == length(d.α) || return false
    tol = sqrt(eps(basefloat(float(eltype(x)))))
    return all(isfinite, x) & (minimum(x) >= zero(eltype(x))) & (abs(sum(x) - one(eltype(x))) <= tol)
end

# `loggamma` throws below zero, so evaluate it at one when the parameters are invalid.
@inline safeshape(valid, α::Number) = select(valid, () -> α, () -> one(α))

function logmvbeta(d::Dirichlet, ::Type{T}) where {T}
    valid = checkparams(d)
    total = zero(T)
    logb = zero(T)
    for αᵢ in d.α
        a = safeshape(valid, convert(T, αᵢ))
        total += a
        logb += loggamma(a)
    end
    value = logb - loggamma(total)
    return select(valid, () -> value, () -> convert(T, NaN))
end

function DensityInterface.logdensityof(d::Dirichlet, x::AbstractVector{<:Number})
    T = float(promote_type(eltype(d.α), eltype(x)))
    length(x) == length(d.α) || return convert(T, -Inf)
    value = -logmvbeta(d, T) + sum(xlogy.(convert.(T, d.α) .- one(T), convert.(T, x)))
    return select(insupport(d, x) & checkparams(d), () -> value, () -> convert(T, -Inf))
end

#=
  Normalized gamma draws are Dirichlet distributed. Working with the log quantiles
  keeps a small shape from underflowing, and `map` rather than a filled vector keeps
  reverse-mode backends, which reject array mutation, on the same path.
=#
function Base.rand(rng::AbstractRNG, d::Dirichlet)
    T = eltype(eltype(d))
    F = basefloat(T)
    valid = checkparams(d)
    logs = map(d.α) do αᵢ
        a = convert(T, αᵢ)
        return select(valid, () -> gammalogquantile(a, rand(rng, F)), () -> convert(T, NaN))
    end
    shift = maximum(logs)
    w = exp.(logs .- shift)
    return w ./ sum(w)
end
