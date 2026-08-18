module WrenBijectorsExt

using Bijectors: Bijectors
using Distributions: Distributions
using Wren: Wren

function Wren.from_unconstrained_vec(dist::Distributions.Distribution)
    return Bijectors.VectorBijectors.from_linked_vec(dist)
end

function Wren.from_vec(dist::Distributions.Distribution)
    return Bijectors.VectorBijectors.from_vec(dist)
end

end # module WrenBijectorsExt
