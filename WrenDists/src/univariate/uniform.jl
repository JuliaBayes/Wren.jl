struct Uniform{A<:Number,B<:Number} <: UnivariateDist
    a::A
    b::B
end

Uniform() = Uniform(0.0, 1.0)

@inline uval(d::Uniform, x::Number) = (x - d.a) / (d.b - d.a)
@inline xval(d::Uniform, u::Number) = muladd(d.b - d.a, u, d.a)

@inline function DensityInterface.logdensityof(d::Uniform, x::Number)
    u = uval(d, x)
    w = oftype(u, d.b - d.a)
    return select((x >= d.a) & (x <= d.b), () -> -log(w), () -> oftype(float(u), -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::Uniform)
    return xval(d, rand(rng, noisetype(d)))
end
