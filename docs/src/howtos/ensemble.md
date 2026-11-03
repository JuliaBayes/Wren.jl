# Ensemble MCMC

[`ensemble`](@ref) samples fixed-dimensional continuous models without gradients.
Load the optional EnsembleMCMC.jl package to enable it. This adapter requires
EnsembleMCMC 0.1; the documentation environment pins a development revision until
that release is registered.

```@example ensemble
using Wren, Wren.Dists, EnsembleMCMC, Random, FlexiChains

@wren function count_model(count)
    rate ~ Exponential()
    @observe count ~ Poisson(rate)
end

walkers = ensemble(Xoshiro(42), count_model(3), 1_000, 4;
    nwalkers=12, warmup=500, progress=false)

size(Array(walkers[1][@vn(rate)]))
```

The arguments request 1,000 retained sweeps from four independent ensembles.
Each ensemble contains 12 coupled walkers. Warmup discards 500 sweeps without
adaptation. Choose the population, warmup and run length for the target.

The result contains one `VNChain` per walker. Each chain has retained sweeps as
rows and independent ensembles as columns. Existing FlexiChains summaries and
diagnostics apply to these individual chains:

```@example ensemble
summarystats(walkers[1])
```

Inspect multiple walkers and independent runs. The walker chains are dependent.
Do not concatenate them as independent chains or sum their ESS values. Posterior
moments may pool all walkers, but ordinary ESS, MCSE and R-hat need the preserved
independent-ensemble columns.

## Initial values and moves

Initialization samples the model distributions using the supplied RNG. For
explicit starts, pass complete model-space `VarNamedTuple` values with axes
(walker, ensemble). A vector supplies the walkers for one ensemble:

```@example ensemble
initial = [VarNamedTuple(rate=r) for r in (0.2, 0.7, 1.2, 2.7)]
move = MoveMixture((StretchMove(), DEMove()), [2, 1])
explicit = ensemble(Xoshiro(43), count_model(3), 100;
    initial_params=initial, move, warmup=50, thin=2, progress=false)
```

Starts must span the unconstrained parameter space and have finite log density.
The default walker count is at least four and meets the chosen move's minimum.
This minimum permits execution; it does not establish good mixing.

Each call advances its RNG. Equal seeds replay a run in a fixed environment.
Model evaluation is serial, and the adapter copies mutable model arguments.
Models must follow [`VecModel`](@ref)'s fixed latent-set and input-ownership rules.

The extras `log_density`, `accepted`, `move_index`, `walker_id` and `sweep`
describe each retained sweep. `log_density` includes the transform Jacobian;
`logjoint`, `logprior` and `loglikelihood` use model-space values. With thinning,
acceptance flags describe retained sweeps, not all intervening proposals.
