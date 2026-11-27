# Times the MNIST multinomial logistic regression model from `bench/adhoc/basic` on the
# full MNIST training set (60,000 images), with Enzyme reverse mode on CPU and with Reactant.
#
# Run from this folder with:
#
#     julia --project=. run.jl

using ADTypes: AutoReactant
using Chairmarks: @be
using Enzyme
using LogExpFunctions: softmax
using MLDatasets: MNIST
using MultivariateStats: fit, PCA, transform
using Printf
using Random: Xoshiro
using Reactant
using Statistics: median
using WrenCore
using WrenDists: WrenDists as WD

# MLDatasets otherwise asks for confirmation before downloading.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# Numbers of features to try. 784 means the raw pixels; anything smaller uses that many PCA
# components.
const DS = (40, 784)
const C = 10 # Number of labels

# Reactant runs on whichever device it picks by default, and its timings include the
# host/device transfer of the parameters and results on each call.
const BACKENDS = (
    ("Enzyme", AutoEnzyme(; mode=Reverse)),
    ("Reactant", AutoReactant(AutoEnzyme(; mode=Reverse))),
)

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

function load_mnist()
    data = MNIST(split=:train)
    nrows, ncols, nimages = size(data.features)
    # D x N matrix of pixel values in [0, 1]
    image_raw = Float64.(reshape(data.features, (nrows * ncols, nimages)))
    # Convert labels from 0-9 to 1-10 for 1-based indexing in Julia
    labels = Int.(data.targets) .+ 1
    return image_raw, labels
end

# Returns an N x D matrix, i.e. one row per image.
function features(image_raw, D)
    D == size(image_raw, 1) && return Matrix(image_raw')
    pca = fit(PCA, image_raw; maxoutdim=D)
    return Matrix(transform(pca, image_raw)')
end

function main()
    image_raw, labels = load_mnist()
    for D in DS
        X = features(image_raw, D)
        N = size(X, 1)
        println(
            "N = $N, D = $D, $(C - 1) * $D + $(C - 1) = $((C - 1) * (D + 1)) parameters",
        )

        model = wren_mnist(X, labels, C)
        reference = nothing
        params = nothing
        for (name, adtype) in BACKENDS
            println("  $name")
            vecmodel = VecModel(model, adtype)
            if params === nothing
                params = vector_parameters(vecmodel, FromPrior(Xoshiro(468)))
            end
            # Call each function once first, so that it's compiled before timing.
            lp(vecmodel, params)
            result = lp_and_grad(vecmodel, params)
            # Check that all backends agree with the first one.
            if reference === nothing
                reference = result
            else
                @assert isapprox(result[1], reference[1]; rtol=1e-8)
                @assert isapprox(result[2], reference[2]; rtol=1e-6)
            end

            b_lp = @be lp($vecmodel, $params)
            b_grad = @be lp_and_grad($vecmodel, $params)
            @printf("    median lp:          %10.3f ms\n", median(b_lp).time * 1e3)
            @printf("    median lp_and_grad: %10.3f ms\n", median(b_grad).time * 1e3)
        end
    end
    return nothing
end

main()
