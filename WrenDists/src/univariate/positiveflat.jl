"""
    PositiveFlat()
    PositiveFlat{T}()

Improper flat distribution over the positive real line, i.e. a log-density of zero for
`x >= 0` and `-Inf` otherwise.

Since this isn't a proper distribution, `rand` can't draw from it. Instead it returns
`exp(u)` where `u` is drawn uniformly from (-2, 2), which is the same way that Stan
initialises positive parameters. The optional type parameter `T` can be used to specify the
sample type, e.g. `PositiveFlat{Float32}()`.

This is like a Stan `real<lower=0>` parameter without a prior.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct PositiveFlat{T<:AbstractFloat} <: PositiveUnivariateDist end

PositiveFlat() = PositiveFlat{Float64}()

# There are no parameters to work out the sample type from, so use `T`.
@inline _promoted_paramtype(::Type{PositiveFlat{T}}) where {T} = T

@inline function DensityInterface.logdensityof(::PositiveFlat, x::Number)
    value = zero(float(x))
    return select(x >= zero(x), () -> value, () -> oftype(value, -Inf))
end

@inline Base.rand(rng::AbstractRNG, ::PositiveFlat{T}) where {T} = exp(rand(rng, Flat{T}()))
