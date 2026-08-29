# [Tutorials](@id tutorials)

## Custom distributions

Sometimes we want to work with models whose log likelihood is not defined based on a typical distribution but instead comes from some arbitrary function.
Here, we'll explain how to write such models in Wren by using hidden Markov models as an example.

### Hidden Markov models

When fitting hidden Markov models, we usually learn about the parameter values by integrating out the latent states using the forward algorithm.
Since we can write any Julia code we'd like inside a `@wren` model, we can rely on [`HiddenMarkovModels.jl`](https://juliastats.org/HiddenMarkovModels.jl/stable/) for HMM structs and the forward algorithm.

Let's simulate a sequence of observations and latent states that we can then fit our model to.

```julia
using HiddenMarkovModels
using Distributions
using LinearAlgebra
using Wren
using Wren.Samplers
using Enzyme
using FillArrays

init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
dists = [MvNormal([-0.5, -0.8], I), MvNormal([0.5, 0.8], I)]

sim_hmm = HMM(init, trans, dists)

T = 200
state_seq, obs_seq = rand(sim_hmm, T)
```

To allow Wren to work with these samples, all we need to define is the `logdensityof` our HMM and a way for it to draw samples from the HMM:

```julia
struct WrappedHMM{T}
    hmm::T
    steps::Int
end

Base.rand(rng::AbstractRNG, d::WrappedHMM) = last(rand(rng, d.hmm, d.steps))

Wren.logdensityof(d::WrappedHMM, obs_seq) = Wren.logdensityof(d.hmm, obs_seq)
```

That's all!

Wren will now be able to be able to work with our custom distribution like it would with any other distribution.
As you can see, we're not required to subtype an abstract type in `Distributions` or conform to any other interface.

Time to see the model in action:

```julia
@wren function hmm_model(steps)
    # we can use p ~ Beta() and 1 - p instead of Dirichlet
    init_1 ~ Beta(2, 2)
    trans_11 ~ Beta(2, 2)
    trans_22 ~ Beta(2, 2)

    # two latent states each emitting two observations
    μ ~ product_distribution(Fill(MvNormal(zeros(2), I), 2))

    # define the HMM
    init = [init_1, 1 - init_1]
    trans = [trans_11 1 - trans_11; 1 - trans_22 trans_22]
    dists = [MvNormal(μ[:, 1], I), MvNormal(μ[:, 2], I)]
    hmm = HMM(init, trans, dists)

    obs_seq ~ WrappedHMM(hmm, steps)

    return nothing
end

# set up the prior model for prior predictive checks
prior_model = hmm_model(T, length(first(obs_seq)))
rand(prior_model)

# condition on the observed sequence and sample
model = prior_model | VarNamedTuple(obs_seq=obs_seq)
chn = nuts(model, 2000, 1)
```

- TODO 1: if we want to sample from several chains, we need to avoid mode-switching if we want parameter recovery to work
- TODO 2: add some plots
- TODO 3: also mention `@addloglikelihood`?
