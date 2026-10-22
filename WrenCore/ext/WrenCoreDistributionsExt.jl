module WrenCoreDistributionsExt

using WrenCore: WrenCore
using Distributions: Distribution, product_distribution

WrenCore.independent(dists::AbstractArray{<:Distribution}) = product_distribution(dists)

end # module WrenCoreDistributionsExt
