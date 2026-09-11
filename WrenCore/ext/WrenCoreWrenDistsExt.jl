module WrenCoreWrenDistsExt

using WrenCore: WrenCore
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

WrenCore.batched(iters, dist::WrenDist; _kwargs...) = WrenCore.iid(dist, size(iters))
WrenCore.batched(iters, n::Normal; _kwargs...) = IIDNormal(size(iters), n.μ, n.σ)
function WrenCore.batched(
    iters,
    ::Val{Normal},
    indexed_mu::WrenCore.Indexed,
    indexed_sigma::WrenCore.Indexed;
    _kwargs...,
)
    return DiagNormal(indexed_mu.data, indexed_sigma.data)
end
function WrenCore.batched(iters, ::Val{Normal}, indexed_mu::WrenCore.Indexed; _kwargs...)
    return IsoNormal(indexed_mu.data)
end

end # module WrenCoreWrenDistsExt
