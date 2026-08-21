module WrenDistributionsExt

using Wren: Wren
using FillArrays: Fill
using Distributions: Distribution, product_distribution

Wren._product(dists::AbstractArray{<:Distribution}) = product_distribution(dists)

Wren.batched(iters, dist::Distribution) = product_distribution(Fill(dist, length(iters)))

end # module WrenDistributionsExt
