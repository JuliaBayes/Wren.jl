# [How-to guides](@id how-to-guides)

## Custom distributions

Sometimes we have to work with models whose log likelihood is not defined based on a typical distribution but instead comes from some arbitrary function.
Here, we'll explain how to write such models in Wren by using hidden Markov models as an example.

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
using Random

init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
dists = [MvNormal([-0.5, -0.8], I), MvNormal([0.5, 0.8], I)]

sim_hmm = HMM(init, trans, dists)

T = 200
state_seq, obs_seq = rand(sim_hmm, T)
```

### Naive model

To allow Wren to work with these samples, all we need to define is the `logdensityof` our HMM and a way for it to draw samples from the HMM[^1]:

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
@wren function hmm_model_naive(steps)
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

prior_model_naive = hmm_model_naive(T)
model_naive = prior_model_naive | VarNamedTuple(obs_seq=obs_seq)
@bmodel model_naive
```

### Optimised model

Typically, the naive model is easiest to reason about because it's closest to how we think of the model conceptually.
Let's see how much extra performance we can squeeze out by moving away from that naive model!

```julia
@wren function hmm_model_fast(steps)
    # we can use p ~ Beta() and 1 - p instead of Dirichlet
    init_1 ~ Beta(2, 2)
    trans_11 ~ Beta(2, 2)
    trans_22 ~ Beta(2, 2)

    # 1, 2 are the first state; 3, 4 are the second
    μ ~ product_distribution(Fill(Normal(), 4))

    # define the HMM
    init = [init_1, 1 - init_1]
    trans = [trans_11 1 - trans_11; 1 - trans_22 trans_22]
    dists = [MvNormal(μ[1:2], I), MvNormal(μ[3:4], I)]
    hmm = HMM(init, trans, dists)

    obs_seq ~ WrappedHMM(hmm, steps)

    return nothing
end

prior_model_fast = hmm_model_fast(T)
model_fast = prior_model_fast | VarNamedTuple(obs_seq=obs_seq)
@bmodel model_fast
```

Unfortunately this didn't speed the model up a lot...
That's because with this model, we spend most of our time doing calculations related to the HMM, making the part we can optimise with Wren's tools relatively small.
We could of course try to speed up the HMM and explore whether [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) improves the situation for us (but that would be a subject for a proper HMM guide).

[^1]: You may also use the [`@addloglikelihood`](@ref) macro to increment the ... well, loglikelihood manually. Be aware that you sacrifice the ability to call `rand` on your model, and that many other parts of the Bayesian workflow will also not integrate as smoothly with models using [`@addloglikelihood`](@ref).
