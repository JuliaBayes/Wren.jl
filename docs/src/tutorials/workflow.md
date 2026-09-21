# Bayesian workflow

Wren provides several functions to enable the Bayesian workflow.

```@example workflow
using Wren
using WrenDists

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    @iid theta ~ Normal(mu, tau) size=J
    sigma ~ Exponential(1)
    @iid y ~ Normal(theta, sigma)
end
```

## Prior predictions

```@example workflow
model_prior = eight_schools(sigma, J)
rand(model_prior)
```

```@example workflow
using FlexiChains
chain_prior = prior(model_prior, 1000, 4; progress=false) 
pushforward_hist(chain_prior, @vn(y))
```

## Conditioning on data

```@example workflow
model_conditioned = model_prior | VarNamedTuple(y=y)
chain_conditioned = nuts(model_conditioned, 1000, 4; progress=false)
summarystats(chain_conditioned)
```

## Posterior predictions

```@example
posterior_predictions = predict(prior_model, chain)
pushforward_hist(chain_conditioned, @vn(y))
```

<!-- TODO once have `fix`, that should also go here -->

