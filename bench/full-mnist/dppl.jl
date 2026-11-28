# Times the DynamicPPL version of the model in `run.jl` on the full MNIST training set
# (60,000 images, raw pixels), with Enzyme reverse mode on CPU.
#
# Run from this folder with:
#
#     julia --project=. dppl.jl

using ADTypes: AutoEnzyme
using Chairmarks: @be
using DifferentiationInterface # Needed for DynamicPPL to use AD backends.
using Distributions
using DynamicPPL
using Enzyme
using LogDensityProblems
using LogExpFunctions: softmax
using MLDatasets: MNIST

# MLDatasets otherwise asks for confirmation before downloading.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@model function dppl_mnist(X, C)
    N, D = size(X)
    W ~ filldist(Normal(), C - 1, D)
    b ~ filldist(Normal(), C - 1)
    nonreference_logits = W * X' .+ b # (C-1) x N
    logits = vcat(zeros(eltype(nonreference_logits), 1, N), nonreference_logits)
    probabilities = softmax(logits; dims=1)
    # `y` is supplied by conditioning the model, not as an argument.
    y = Vector{Int}(undef, N)
    for i in 1:N
        y[i] ~ Categorical(probabilities[:, i])
    end
end

data = MNIST(split=:train)
# N x 784 matrix of pixel values in [0, 1], one row per image
X = Matrix(Float64.(reshape(data.features, 28 * 28, :))')
# Convert labels from 0-9 to 1-10 for 1-based indexing in Julia
y = data.targets .+ 1

model = dppl_mnist(X, 10) | (; y=y)

adtype = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
ldf = DynamicPPL.LogDensityFunction(
    model,
    DynamicPPL.getlogjoint_internal,
    DynamicPPL.LinkAll();
    adtype,
)
params = randn(LogDensityProblems.dimension(ldf))
# Call each function once first, so that it's compiled before timing.
LogDensityProblems.logdensity(ldf, params)
LogDensityProblems.logdensity_and_gradient(ldf, params)
display(@be LogDensityProblems.logdensity($ldf, $params))
display(@be LogDensityProblems.logdensity_and_gradient($ldf, $params))
