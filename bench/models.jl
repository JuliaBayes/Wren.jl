module BenchmarkModels

using DelimitedFiles
using Distributions
using LinearAlgebra: I
using LogExpFunctions: softmax
using Wren: Wren, VarNamedTuple
using DynamicPPL: DynamicPPL, filldist

Wren.@model function wren_normal()
    x ~ Normal()
    return nothing
end

DynamicPPL.@model function dppl_normal()
    x ~ Normal()
    return nothing
end

const EIGHT_SCHOOLS_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SCHOOLS_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
const EIGHT_SCHOOLS_J = 8

Wren.@model function wren_eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ MvNormal(fill(mu, J), tau^2 * I)
    for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

DynamicPPL.@model function dppl_eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ MvNormal(fill(mu, J), tau^2 * I)
    for i in 1:J
        y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

path = "mnist.csv"
data, _ = readdlm(path, ',', header = true)
labels = Int.(data[:, 1])
image_vec = data[:, 2:end]
C = 10 # Number of labels
D = size(image_vec, 2)

Wren.@model function wren_mnist(X, y, C)
    N, D = size(X)
    W ~ filldist(Normal(), C - 1, D)
    b ~ filldist(Normal(), C - 1)
    nonreference_logits = W * X' .+ b # (C-1) x N
    logits = vcat(zeros(eltype(nonreference_logits), 1, N), nonreference_logits)
    probabilities = softmax(logits; dims=1)
    for i in 1:N
        @observe y[i] ~ Categorical(probabilities[:, i])
    end
end

DynamicPPL.@model function dppl_mnist(X, y, C)
    N, D = size(X)
    W ~ filldist(Normal(), C - 1, D)
    b ~ filldist(Normal(), C - 1)
    nonreference_logits = W * X' .+ b
    logits = vcat(zeros(eltype(nonreference_logits), 1, N), nonreference_logits)
    probabilities = softmax(logits; dims=1)
    for i in 1:N
        y[i] ~ Categorical(probabilities[:, i])
    end
end

function models()
    return (
        (name="normal", wren=wren_normal(), dppl=dppl_normal()),
        (name="eight schools", wren=wren_eight_schools(EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, EIGHT_SCHOOLS_J), dppl=dppl_eight_schools(EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, EIGHT_SCHOOLS_J)),
        (name="mnist", wren=wren_mnist(image_vec, labels, C), dppl=dppl_mnist(image_vec, labels, C))
    )
end

end # module BenchmarkModels
