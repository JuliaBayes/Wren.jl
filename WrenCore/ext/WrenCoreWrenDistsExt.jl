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
    IsoNormal

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

end # module WrenCoreWrenDistsExt
