# Bayesian workflow

Wren provides several functions to enable the Bayesian workflow.

<!-- once we have HalfNormal or HalfCauchy, we could switch this to WrenDists -->

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

# Save the mapping from species index (integer) to species name (string)
species_names = sort(unique(penguins[!, [:species, :species_idx]]), :species_idx).species

# Define a model for penguin bill length as a function of species and body mass
@wren function bill_model(bill_length, species, body_mass)
    n_species = length(unique(species))
    @iid beta1 ~ Normal(0, 1) size=n_species
    beta2 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    mu = beta1[species] .+ beta2 .* body_mass
    @batch for i in eachindex(bill_length)
        bill_length[i] ~ Normal(mu[i], sigma)
    end
end

model = bill_model(penguins.bill_length_mm, penguins.species_idx, penguins.body_mass_g)
```

<!-- Should we make a note about not being able to use `@observe` in conjunction with `condition`? Or did I confuse myself again and we actually can... idk. I got an error just now. -->

## Prior predictions

<!-- Model setup and single prior draw -->

```@example workflow
rand(model)
```

<!-- Sampling an entire chain from the prior, checking summary stats, plotting the result using hist with nested quantile -->

```@example workflow
chain_prior = prior(model, 1000, 4; progress=false)
summarystats(chain_prior)
```

```@example workflow
using FlexiChains
pushforward_hist(chain_prior, @vn(y))
```

## Conditioning on data

```@example workflow
import Enzyme
chain_posterior = nuts(model | VarNamedTuple(y=y), 1000, 4; progress=false)
summarystats(chain_posterior)
```

## Posterior predictions

```@example
posterior_predictions = predict(model, chain)
pushforward_hist(chain_posterior, @vn(y))
```

<!-- TODO once have `fix`, that should also go here -->

