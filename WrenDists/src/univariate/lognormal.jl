"""
    LogNormal(μ, σ)
    LogNormal(μ)
    LogNormal()

Log-normal distribution with log-scale mean `μ` (default 0.0) and log-scale standard
deviation `σ` (default 1.0).

**Customised batched implementations**: Yes
"""
struct LogNormal{M<:Number,S<:Number} <: PositiveUnivariateDist
    μ::M
    σ::S
end

LogNormal(μ) = LogNormal(μ, one(μ))
LogNormal() = LogNormal(0.0, 1.0)

@inline zval(d::LogNormal, x::Number) = (log(x) - d.μ) / d.σ
@inline xval(d::LogNormal, z::Number) = exp(muladd(d.σ, z, d.μ))

@inline function DensityInterface.logdensityof(d::LogNormal, x::Number)
    lx = log(select(x > zero(x), () -> x, () -> one(x)))
    z = (lx - d.μ) / d.σ
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return select(
        x > zero(x),
        () -> -(z^2 + log2π) / 2 - log(σ) - lx,
        () -> oftype(float(z), -Inf),
    )
end

# Apply `μ` and `σ` after drawing noise so automatic differentiation can follow them.
@inline Base.rand(rng::AbstractRNG, d::LogNormal) = xval(d, randn(rng, noisetype(d)))

# --- Batched LogNormal ---

struct BatchedLogNormal{M<:AbstractArray,S<:AbstractArray} <: BatchedUnivariate
    μ::M
    σ::S
end
Base.size(d::BatchedLogNormal) = size(d.μ)
function Base.rand(rng::AbstractRNG, d::BatchedLogNormal)
    return exp.(randn(rng, noisetype(d), size(d)) .* d.σ .+ d.μ)
end

function _lognormal_loop(mu, sigma, x)
    T = float(promote_type(eltype(x), eltype(mu), eltype(sigma)))
    s = zero(T)
    for i in eachindex(x, mu, sigma)
        lx = log(x[i])
        z = (lx - mu[i]) / sigma[i]
        s -= abs2(z) / 2 + log(sigma[i]) + lx
    end
    s
end

EnzymeRules.@easy_rule(
    _lognormal_loop(mu, sigma, x),
    @setup(z = (log.(x) .- mu) ./ sigma),
    (z ./ sigma, (abs2.(z) .- 1) ./ sigma, -(z ./ (x .* sigma) .+ inv.(x))),
)

function DensityInterface.logdensityof(d::BatchedLogNormal, x)
    # Benchmarks @ https://github.com/JuliaBayes/Wren.jl/issues/106
    mu, sigma = d.μ, d.σ
    lognorm = length(x) * convert(float(eltype(x)), log2π) / 2
    return if within_compile()
        safe_x = ifelse.(x .> 0, x, one.(x))
        value =
            sum(@. -(abs2((log(safe_x) - mu) / sigma)) / 2 - log(sigma) - log(safe_x)) -
            lognorm
        ifelse(all(x .> 0), value, oftype(value, -Inf))
    else
        T = promote_type(float(eltype(x)), eltype(mu), float(eltype(sigma)))
        # Keep unsupported data out of the kernel and its Enzyme rule.
        value = all(>(0), x) ? _lognormal_loop(mu, sigma, x) : convert(T, -Inf)
        value - lognorm
    end
end
Plaice.scalar_to_scalar_bijector(::BatchedLogNormal) = Plaice.Log(0.0, 1)
