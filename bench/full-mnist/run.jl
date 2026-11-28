# Times the MNIST multinomial logistic regression model from `bench/adhoc/basic` on the
# full MNIST training set (60,000 images, raw pixels), with Enzyme reverse mode on CPU and
# with Reactant.
#
# Run from this folder with:
#
#     julia --project=. run.jl

using ADTypes: AutoReactant
using Chairmarks: @be
using Enzyme
using LogExpFunctions: softmax
using MLDatasets: MNIST
using Reactant
using WrenCore
using WrenDists: WrenDists as WD

# MLDatasets otherwise asks for confirmation before downloading.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# Same model as `bench/adhoc/basic/models.jl`.
WrenCore.@wren function wren_mnist(X, y, C)
    N, D = size(X)
    @iid W ~ WD.Normal() size=(C-1, D)
    @iid b ~ WD.Normal() size=(C-1)
    nonreference_logits = W * X' .+ b # (C-1) x N
    logits = vcat(zeros(eltype(nonreference_logits), 1, N), nonreference_logits)
    probabilities = softmax(logits; dims=1)
    @batch for i in 1:N
        @observe y[i] ~ WD.Categorical(probabilities[:, i])
    end
end

data = MNIST(split=:train)
# N x 784 matrix of pixel values in [0, 1], one row per image
X = Matrix(Float64.(reshape(data.features, 28 * 28, :))')
# Convert labels from 0-9 to 1-10 for 1-based indexing in Julia
y = data.targets .+ 1

model = wren_mnist(X, y, 10)

# Reactant runs on whichever device it picks by default, and its timings include the
# host/device transfer of the parameters and results on each call.
for adtype in (AutoEnzyme(; mode=Reverse), AutoReactant(AutoEnzyme(; mode=Reverse)))
    vecmodel = VecModel(model, adtype)
    params = vector_parameters(vecmodel)
    # Call each function once first, so that it's compiled before timing.
    lp(vecmodel, params)
    lp_and_grad(vecmodel, params)
    println(adtype)
    display(@be lp($vecmodel, $params))
    display(@be lp_and_grad($vecmodel, $params))
end
