"""
    Void

A latent source that provides no values (it errors when asked for any).
"""
struct Void <: AbstractLatentSource end

function get_latent(::Void, vn, dist, sitename)
    throw(ArgumentError("No latent value provided for variable $vn"))
end

scope(::Void, vn) = Void()
