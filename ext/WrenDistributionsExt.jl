module WrenDistributionsExt

using Wren: Wren
using Distributions: Distribution, product_distribution

Wren._product(dists::AbstractArray{<:Distribution}) = product_distribution(dists)

end # module WrenDistributionsExt
