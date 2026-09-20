module WrenCoreWrenDistsExt

using WrenCore: WrenCore
using FillArrays: Fill
using WrenDists:
    WrenDist,
    UnivariateDist,
    MultivariateDist,
    Independent,
    Repeated,
    Normal,
    IIDNormal,
    DiagNormal,
    IsoNormal,
    Bernoulli,
    BatchedBernoulli,
    Binomial,
    BatchedBinomial,
    Categorical,
    BatchedCategorical,
    Poisson,
    BatchedPoisson,
    Exponential,
    BatchedExponential,
    LogNormal,
    BatchedLogNormal,
    TDist,
    BatchedTDist

# Generic fallbacks
WrenCore.iid(dist::WrenDist, dims::Dims) = Independent(Repeated(dist, dims))
WrenCore.independent(dists::AbstractArray{<:UnivariateDist}) = Independent(dists)
function WrenCore.independent(dists::AbstractArray{<:MultivariateDist})
    throw(
        ArgumentError(
            "independent products of multivariate distributions are not supported; " *
            "use `iid` for identically distributed draws or a single multivariate measure.",
        ),
    )
end

# Normal(consts...)
WrenCore.iid(dist::Normal, size::Dims) = IIDNormal(size, dist.μ, dist.σ)

# Normal(x[i], sigma[i]) -> DiagNormal(x, sigma)
function WrenCore.batched(
    iters,
    ::Val{Normal},
    indexed_mu::WrenCore.Indexed,
    indexed_sigma::WrenCore.Indexed;
    _kwargs...,
)
    return DiagNormal(indexed_mu.data, indexed_sigma.data)
end

# Normal(x[i], const_sigma) -> IsoNormal(x, const_sigma)
function WrenCore.batched(
    iters,
    ::Val{Normal},
    indexed_mu::WrenCore.Indexed,
    sigma::Number;
    _kwargs...,
)
    return IsoNormal(indexed_mu.data, sigma)
end

# Normal(const_mu, sigma[i]) -> DiagNormal(Fill(const_mu, ...), sigma)
function WrenCore.batched(
    iters,
    ::Val{Normal},
    mu::Number,
    indexed_sigma::WrenCore.Indexed;
    _kwargs...,
)
    return DiagNormal(Fill(mu, size(indexed_sigma.data)), indexed_sigma.data)
end

# Normal(x[i]) --> IsoNormal(x, 1.0)
function WrenCore.batched(iters, ::Val{Normal}, indexed_mu::WrenCore.Indexed; _kwargs...)
    return IsoNormal(indexed_mu.data, one(eltype(indexed_mu.data)))
end

# Single-parameter distributions: Bernoulli(p[i]), Poisson(lambda[i]), Exponential(theta[i])
for (D, B) in (
    (:Bernoulli, :BatchedBernoulli),
    (:Poisson, :BatchedPoisson),
    (:Exponential, :BatchedExponential),
)
    @eval function WrenCore.batched(iters, ::Val{$D}, indexed::WrenCore.Indexed; _kwargs...)
        return $B(indexed.data)
    end
end

# LogNormal(mu[i]) -> BatchedLogNormal(mu, Fill(1, ...))
function WrenCore.batched(iters, ::Val{LogNormal}, indexed_mu::WrenCore.Indexed; _kwargs...)
    return BatchedLogNormal(
        indexed_mu.data,
        Fill(one(eltype(indexed_mu.data)), size(indexed_mu.data)),
    )
end

# Binomial(n[i]) -> BatchedBinomial(n, Fill(0.5, ...))
function WrenCore.batched(iters, ::Val{Binomial}, indexed_n::WrenCore.Indexed; _kwargs...)
    return BatchedBinomial(indexed_n.data, Fill(0.5, size(indexed_n.data)))
end

# Two-parameter distributions: Binomial(n, p) and LogNormal(mu, sigma). Constant arguments
# are stored as `Fill` arrays matching the shape of the indexed argument.
for (D, B) in ((:Binomial, :BatchedBinomial), (:LogNormal, :BatchedLogNormal))
    @eval begin
        function WrenCore.batched(
            iters,
            ::Val{$D},
            a::WrenCore.Indexed,
            b::WrenCore.Indexed;
            _kwargs...,
        )
            return $B(a.data, b.data)
        end
        function WrenCore.batched(
            iters,
            ::Val{$D},
            a::WrenCore.Indexed,
            b::Number;
            _kwargs...,
        )
            return $B(a.data, Fill(b, size(a.data)))
        end
        function WrenCore.batched(
            iters,
            ::Val{$D},
            a::Number,
            b::WrenCore.Indexed;
            _kwargs...,
        )
            return $B(Fill(a, size(b.data)), b.data)
        end
    end
end

# TDist: three parameters (ν, μ, σ). Constant arguments become Fill arrays.
# TDist(ν, μ[i], σ[i])
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::Number,
    μ::WrenCore.Indexed,
    σ::WrenCore.Indexed;
    _kwargs...,
)
    return BatchedTDist(Fill(ν, size(μ.data)), μ.data, σ.data)
end
# TDist(ν, μ[i], σ)
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::Number,
    μ::WrenCore.Indexed,
    σ::Number;
    _kwargs...,
)
    return BatchedTDist(Fill(ν, size(μ.data)), μ.data, Fill(σ, size(μ.data)))
end
# TDist(ν, μ, σ[i])
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::Number,
    μ::Number,
    σ::WrenCore.Indexed;
    _kwargs...,
)
    return BatchedTDist(Fill(ν, size(σ.data)), Fill(μ, size(σ.data)), σ.data)
end
# TDist(ν[i], μ[i], σ[i])
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::WrenCore.Indexed,
    μ::WrenCore.Indexed,
    σ::WrenCore.Indexed;
    _kwargs...,
)
    return BatchedTDist(ν.data, μ.data, σ.data)
end
# TDist(ν[i], μ[i], σ)
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::WrenCore.Indexed,
    μ::WrenCore.Indexed,
    σ::Number;
    _kwargs...,
)
    return BatchedTDist(ν.data, μ.data, Fill(σ, size(μ.data)))
end
# TDist(ν[i], μ, σ[i])
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::WrenCore.Indexed,
    μ::Number,
    σ::WrenCore.Indexed;
    _kwargs...,
)
    return BatchedTDist(ν.data, Fill(μ, size(ν.data)), σ.data)
end
# TDist(ν[i], μ, σ)
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::WrenCore.Indexed,
    μ::Number,
    σ::Number;
    _kwargs...,
)
    return BatchedTDist(ν.data, Fill(μ, size(ν.data)), Fill(σ, size(ν.data)))
end
# TDist(ν, μ[i]) -> σ defaults to 1
function WrenCore.batched(iters, ::Val{TDist}, ν::Number, μ::WrenCore.Indexed; _kwargs...)
    T = eltype(μ.data)
    return BatchedTDist(Fill(ν, size(μ.data)), μ.data, Fill(one(T), size(μ.data)))
end
# TDist(ν[i], μ[i])
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::WrenCore.Indexed,
    μ::WrenCore.Indexed;
    _kwargs...,
)
    T = eltype(μ.data)
    return BatchedTDist(ν.data, μ.data, Fill(one(T), size(μ.data)))
end
# TDist(ν[i]) -> μ=0, σ=1
function WrenCore.batched(iters, ::Val{TDist}, ν::WrenCore.Indexed; _kwargs...)
    T = float(eltype(ν.data))
    return BatchedTDist(ν.data, Fill(zero(T), size(ν.data)), Fill(one(T), size(ν.data)))
end

# Categorical
function WrenCore.batched(
    iters,
    ::Val{Categorical},
    indexed::WrenCore.Indexed{1,1,<:AbstractVector{<:AbstractVector}};
    _kwargs...,
)
    # indexed.data[i] is a vector of probabilities, so indexed.data is a vector (batch_size)
    # of vectors (categories), and stack(indexed.data) will be categories x batch_size
    return BatchedCategorical(stack(indexed.data))
end
function WrenCore.batched(
    iters,
    ::Val{Categorical},
    indexed::WrenCore.Indexed{1,2,<:AbstractMatrix};
    _kwargs...,
)
    # indexed.data has dimensions (batch_size, categories), so we need to transpose
    return BatchedCategorical(permutedims(indexed.data))
end
function WrenCore.batched(
    iters,
    ::Val{Categorical},
    indexed::WrenCore.Indexed{2,2,<:AbstractMatrix};
    _kwargs...,
)
    # indexed.data has dimensions (categories, batch_size), so we are good
    return BatchedCategorical(indexed.data)
end

end # module WrenCoreWrenDistsExt
