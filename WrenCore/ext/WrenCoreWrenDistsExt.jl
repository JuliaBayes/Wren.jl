module WrenCoreWrenDistsExt

using WrenCore: WrenCore
using FillArrays: Fill
using WrenDists:
    BatchedBernoulli,
    BatchedBernoulliLogit,
    BatchedBeta,
    BatchedBinomial,
    BatchedBinomialLogit,
    BatchedCategorical,
    BatchedExponential,
    BatchedGamma,
    BatchedLogNormal,
    BatchedNegativeBinomial,
    BatchedPoisson,
    BatchedPoissonLog,
    BatchedTDist,
    BatchedWeibull,
    Bernoulli,
    BernoulliLogit,
    Beta,
    Binomial,
    BinomialLogit,
    Categorical,
    DiagNormal,
    Exponential,
    Gamma,
    IIDNormal,
    IIDTDist,
    Independent,
    IsoNormal,
    LogNormal,
    MultivariateDist,
    NegativeBinomial,
    Normal,
    Poisson,
    PoissonLog,
    Repeated,
    TDist,
    UnivariateDist,
    Weibull,
    WrenDist

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

# TDist(consts...)
WrenCore.iid(dist::TDist, size::Dims) = IIDTDist(size, dist.ν, dist.μ, dist.σ)

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
    return DiagNormal(Fill(mu, size(iters)), indexed_sigma.data)
end

# Normal(x[i]) --> IsoNormal(x, 1.0)
function WrenCore.batched(iters, ::Val{Normal}, indexed_mu::WrenCore.Indexed; _kwargs...)
    return IsoNormal(indexed_mu.data, one(eltype(indexed_mu.data)))
end

# Single-parameter distributions: Bernoulli(p[i]), Poisson(lambda[i]), Exponential(theta[i]),
# and the logit/log-rate forms BernoulliLogit(logitp[i]), PoissonLog(loglambda[i])
for (D, B) in (
    (:Bernoulli, :BatchedBernoulli),
    (:BernoulliLogit, :BatchedBernoulliLogit),
    (:Poisson, :BatchedPoisson),
    (:PoissonLog, :BatchedPoissonLog),
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
        Fill(one(eltype(indexed_mu.data)), size(iters)),
    )
end

# Binomial(n[i]) -> BatchedBinomial(n, Fill(0.5, ...))
function WrenCore.batched(iters, ::Val{Binomial}, indexed_n::WrenCore.Indexed; _kwargs...)
    return BatchedBinomial(indexed_n.data, Fill(0.5, size(iters)))
end

# BinomialLogit(n[i]) -> BatchedBinomialLogit(n, Fill(0.0, ...))
function WrenCore.batched(
    iters,
    ::Val{BinomialLogit},
    indexed_n::WrenCore.Indexed;
    _kwargs...,
)
    return BatchedBinomialLogit(indexed_n.data, Fill(0.0, size(iters)))
end

function WrenCore.batched(
    iters,
    ::Val{NegativeBinomial},
    indexed_r::WrenCore.Indexed;
    _kwargs...,
)
    return BatchedNegativeBinomial(indexed_r.data, Fill(0.5, size(iters)))
end

# One-parameter calls with unit scale.
for (D, B) in ((:Gamma, :BatchedGamma), (:Weibull, :BatchedWeibull))
    @eval function WrenCore.batched(iters, ::Val{$D}, indexed::WrenCore.Indexed; _kwargs...)
        return $B(indexed.data, Fill(one(eltype(indexed.data)), size(iters)))
    end
end

# Two-parameter distributions store constant arguments as `Fill` arrays matching the
# shape of the indexed argument.
for (D, B) in (
    (:Binomial, :BatchedBinomial),
    (:BinomialLogit, :BatchedBinomialLogit),
    (:LogNormal, :BatchedLogNormal),
    (:Gamma, :BatchedGamma),
    (:Beta, :BatchedBeta),
    (:Weibull, :BatchedWeibull),
    (:NegativeBinomial, :BatchedNegativeBinomial),
)
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
            return $B(a.data, Fill(b, size(iters)))
        end
        function WrenCore.batched(
            iters,
            ::Val{$D},
            a::Number,
            b::WrenCore.Indexed;
            _kwargs...,
        )
            return $B(Fill(a, size(iters)), b.data)
        end
    end
end

# TDist: three parameters (ν, μ, σ). Constant arguments become Fill arrays.
const _TDistArg = Union{WrenCore.Indexed,Number}
_to_array(x::WrenCore.Indexed, s) = x.data
_to_array(x::Number, s) = Fill(x, s)
function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::Tν,
    μ::Tμ,
    σ::Tσ;
    _kwargs...,
) where {Tν<:_TDistArg,Tμ<:_TDistArg,Tσ<:_TDistArg}
    s = size(iters)
    return BatchedTDist(_to_array(ν, s), _to_array(μ, s), _to_array(σ, s))
end

function WrenCore.batched(
    iters,
    ::Val{TDist},
    ν::Tν,
    μ::Tμ;
    _kwargs...,
) where {Tν<:_TDistArg,Tμ<:_TDistArg}
    s = size(iters)
    T = eltype(_to_array(μ, s))
    return BatchedTDist(_to_array(ν, s), _to_array(μ, s), Fill(one(T), s))
end

function WrenCore.batched(iters, ::Val{TDist}, ν::Tν; _kwargs...) where {Tν<:_TDistArg}
    s = size(iters)
    T = float(eltype(_to_array(ν, s)))
    return BatchedTDist(_to_array(ν, s), Fill(zero(T), s), Fill(one(T), s))
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
