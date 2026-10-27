"""
    Flat()
    Flat{T}()

Improper flat distribution over the real line, i.e. a log-density of zero everywhere.

This is mostly useful as a prior when you want the posterior to be determined by the
likelihood alone (like a parameter without a prior in Stan).

Since this isn't a proper distribution, `rand` can't draw from it. Instead it returns a
value drawn uniformly from (-2, 2), which is the same range that Stan uses to initialise
unconstrained parameters.

Since there are no parameters to take the type of samples from, it is given by `T`, which
defaults to `Float64`.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct Flat{T<:AbstractFloat} <: UnconstrainedUnivariateDist end

Flat() = Flat{Float64}()

# There are no parameters to work out the sample type from, so use `T`.
@inline _promoted_paramtype(::Type{Flat{T}}) where {T} = T

@inline DensityInterface.logdensityof(::Flat, x::Number) = zero(float(x))

@inline Base.rand(rng::AbstractRNG, ::Flat{T}) where {T} = 4 * rand(rng, T) - 2
