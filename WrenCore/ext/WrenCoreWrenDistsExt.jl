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
    BatchedLogNormal

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
