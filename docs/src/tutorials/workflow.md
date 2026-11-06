# Bayesian workflow

In the worked example below, we demonstrate how Wren and the JuliaBayes ecosystem enable the [Bayesian Workflow](https://avehtari.github.io/Bayesian-Workflow/).

We'll concoct an example with the [Palmer penguins dataset](https://github.com/devmotion/PalmerPenguins.jl).
This dataset includes measurements of three different Antarctic penguins species.

We first have to prepare the dataset.
We'll z-standardise the variables because it usually improves sampling efficiency and simplifies choosing priors.

```@example workflow
using Wren, PalmerPenguins, DataFrames, FlexiChains
using WrenDists: Normal, Exponential
using StatsBase: denserank, fit, ZScoreTransform, reconstruct

# Load data
penguins = DataFrame(PalmerPenguins.load())

# Drop missing values first (this ensures that standardisation doesn't trip over)
dropmissing!(penguins)

# Fit standardisation transforms so we can unstandardise predictions later
bill_zs = fit(ZScoreTransform, Float64.(penguins.bill_length_mm))
mass_zs = fit(ZScoreTransform, Float64.(penguins.body_mass_g))

# Tidy up the data
standardize(x) = (x .- mean(x)) ./ std(x)
transform!(penguins, names(penguins, Real) .=> standardize => identity)
transform!(penguins, :species => denserank => :species_idx)
nothing # hide
```

Let's say we're interested in learning if a penguin's bill length increases with body mass, while taking into account that bill lengths between species might differ, too.
The model below uses [`@iid`](@ref iid) for the per-species intercepts and a [`@batch`](@ref batched) loop for the likelihood:

```@example workflow
# Define a model for penguin bill length as a function of species and body mass
@wren function bill_model(bill_length, species, body_mass)
    n_species = maximum(species)
    @iid beta1 ~ Normal(0, 1) size=n_species
    beta2 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    mu = beta1[species] .+ beta2 .* body_mass
    @batch for i in eachindex(bill_length)
        bill_length[i] ~ Normal(mu[i], sigma)
    end
end
```

## Prior predictions

We can now instantiate the Wren model and use it with `rand(model)` to draw from the prior (see [Working with models](@ref models)):

```@example workflow
model = bill_model(penguins.bill_length_mm, penguins.species_idx, penguins.body_mass_g)
rand(model)
```

!!! note

    You may have noticed that we did not use `@observe` in our model definition.
    This is because `@observe param ~ dist` will always treat `param` as an observed variable, even if it was not conditioned on.
    The consequence would be that we could draw neither prior nor posterior predictions for it.

Alternatively, we can use [`prior`](@ref) to create a [`FlexiChain`](@extref FlexiChains.FlexiChain) that contains as many prior draws as we specify, and summarise it with [`summarystats`](@extref FlexiChains :jl:method:`StatsBase.summarystats-Tuple{FlexiChains.FlexiChain}`).

```@example workflow
chain_prior = prior(model, 1000, 4; progress=false)
summarystats(chain_prior[[@vn(beta1), @vn(beta2), @vn(sigma)]])
```

While we could technically continue working with the chain, the `rand(model)` syntax lends itself nicely towards visualising individual samples from the prior distribution.

```@example workflow
using CairoMakie

colors = Makie.wong_colors()
fig = Figure()
ax = Axis(fig[1, 1], xlabel="body mass (g)", ylabel="bill length (mm)")

xs_z = -3:0.1:3

# Draw 20 prior predictions for the slope of body mass
for _ in 1:20
    vnt = rand(model)
    ys_z = vnt[@vn(beta2)] * xs_z
    ys = reconstruct(bill_zs, ys_z)
    xs = reconstruct(mass_zs, xs_z)
    lines!(xs, ys, color=(colors[1], 0.6))
end

fig
```

You can read more about working with [`VarName`](@extref VarNames.VarName) and [`VarNamedTuple`](@extref VarNames.VarNamedTuple) in the [VarNames.jl docs](https://github.com/JuliaBayes/VarNames.jl).

We may also want to check if the prior distribution of `bill_length` looks reasonable, i.e., that it doesn't include impossible values such as negative `bill_length`.
FlexiChains provides helpful functions here – first, we reconstruct the draws on the millimeter scale with [`transform_values`](@extref FlexiChains.transform_values) and then visualise the prior predictive distribution with [`pushforward_hist`](@extref FlexiChains.Makie.pushforward_hist):

```@example workflow
using FlexiChains: transform_values
chain_prior =
    transform_values(chain_prior, @vn(bill_length) => (v -> reconstruct(bill_zs, v)))
observed = reconstruct(bill_zs, penguins.bill_length_mm)
FlexiChains.Makie.pushforward_hist(
    chain_prior,
    @vn(bill_length);
    observed=observed,
    axis=(; xlabel="bill_length_mm", limits=((0, 100), nothing)),
)
```

## Conditioning on data

Having had a look at the prior, we're now ready to condition the model on the observed data and sample from the posterior.
Wren's default autodiff backend is Enzyme (see [Working with Enzyme](@ref enzyme)), so we need to make sure to import that first.
Then we condition with `|` (see [`condition`](@ref)) and sample with [`nuts`](@ref):

```@example workflow
import Enzyme
model_cond = model | VarNamedTuple(bill_length=penguins.bill_length_mm)
chain_post = nuts(model_cond, 1000, 4; progress=false)
summarystats(chain_post)
```

Normally we would like to check that chains mixed and sampling went well with more diagnostics, such as diagnostic plots `plot(chain_post)` or checking for divergent transitions `sum(chain_post[:is_divergent]) / prod(size(chain_post))`, but we'll leave it at mentioning these for this simple model.

## Posterior predictions

Towards the end of the Bayesian workflow, we typically want to check if the model approximates the data-generating process reasonably well with a posterior predictive check.

A posterior predictive check uses the fitted parameters and values of the covariates `species` and `body_mass` to predict new values of the outcome `bill_length`.
This can be done with either the same covariate values used to fit the model, or with a new set of covariate values.
A new set could either represent a hypothetical new group of penguins that we want to predict for, or help us visualise how `bill_length` changes with `body_mass`, resulting in a plot similar to the prior slope visualisation above.

We'll begin with the former and use [`predict`](@ref) to predict for the same covariates we fit the model with.
`FlexiChains.pushforward_hist` then plots histograms for our posterior prediction and overlays the observed data, giving us an opportunity to spot if the two would clearly diverge:

```@example workflow
chain_pred = predict(model, chain_post)
chain_pred =
    transform_values(chain_pred, @vn(bill_length) => (v -> reconstruct(bill_zs, v)))
FlexiChains.Makie.pushforward_hist(
    chain_pred,
    @vn(bill_length);
    observed=observed,
    axis=(; xlabel="bill length (mm)"),
)
```

The model seems to capture the distribution of the observed data well!

Now let's see what the model learned about how `body_mass` relates to `bill_length`, by drawing single samples from the chain and extracting them with [`parameters`](@extref VarNames.parameters).

```@example workflow
fig = Figure()
ax = Axis(fig[1, 1], xlabel="body mass (g)", ylabel="bill length (mm)")

xs_z = -3:0.1:3

# Draw 20 posterior samples of the slope of body mass
for _ in 1:20
    vnt = parameters(rand(chain_post))
    ys_z = vnt[@vn(beta2)] * xs_z
    ys = reconstruct(bill_zs, ys_z)
    xs = reconstruct(mass_zs, xs_z)
    lines!(xs, ys, color=(colors[1], 0.6))
end

fig
```

As we may have expected, more penguin means more bill!

## Generated quantities

Most of the time in applied statistical modelling, we're after a quantity that we can't express as a single parameter of our model.
Rather, it's often some downstream quantity that we compute by combining several parameters with (often new) observations.
These downstream quantities are typically called generated quantities.

Continuing with the penguin example, let's assume that there is something special about the conditions on each island.
These conditions make _all_ the penguins on a given island more similar to each other, because there might be better food sources near some islands than others.
And of course, all penguin species would benefit from more fish (at least... I would think all penguins like fish?! And krill of course, ok).

We can express this nestedness in Bayesian modelling with a hierarchical model.
Since this model component is quite common and thus reusable, it's a good candidate to be written as a [submodel](@ref):

```@example workflow
@wren function hierarchical(size)
    mu ~ Normal()
    tau ~ Exponential(1)
    @iid z ~ Normal() size=size
    return mu .+ tau * z
end
```

Assume we're mostly interested in the contrast between species 1 and 2.
We could then consider dropping species 3 before fitting the model, but we still have reason to keep species 3 as it'll contribute to estimating `beta2` and `beta3` better.

```@example workflow
@wren function bill_model_hierarchical(bill_length, island, species, body_mass)
    n_species = maximum(species)
    n_islands = maximum(island)

    @iid beta1 ~ Normal(0, 1) size=n_species
    beta2 ~ Normal(0, 1)
    beta3 = (island_effects ~ hierarchical(n_islands))
    sigma ~ Exponential(1)
    
    mu = beta1[species] .+ beta2 .* body_mass .+ beta3[island]
    @batch for i in eachindex(bill_length)
        bill_length[i] ~ Normal(mu[i], sigma)
    end

    # TODO what would be a good contrast here?
    @gq contrast = ...
end
```

TODO Simon: 
- describe the GQ more
- describe that we now first fit the model with the data we observed

```@example workflow
transform!(penguins, :island => denserank => :island_idx)
model = bill_model_hierarchical(
  penguins.bill_length_mm,
  penguins.island_idx,
  penguins.species_idx,
  penguins.body_mass_g,
)
chain = nuts(model | VarNamedTuple(bill_length=penguins.bill_length_mm), 1000, 4; progress=false)
```

TODO Simon: 
- describe the estimate that we're after (contrast on hypothetical islands)

```@example workflow
n_new = 200
island_new = 1:n_new
species_new = repeat([1, 2], div(n_new, 2))
# Not sure about this one, need to think again what we should draw here
body_mass_g_new = rand(penguins.body_mass_g, n_new)
```

TODO Simon: 
- describe why subsetting the chain gets us where we want to be here

```@example workflow
chain_sub = chain[filter(!=(@vn(island_effects.z)), FlexiChains.parameters(chain))]
model_new = bill_model_hierarchical(zeros(n_new), island_new, species_new, body_mass_g_new)
chain_new = predict(model_new, chain_sub)
```
