# Bayesian workflow

Wren provides several functions to enable the Bayesian workflow.

<!-- once we have HalfNormal or HalfCauchy, we could switch this to WrenDists -->

```@example workflow
using Wren
using Distributions

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    @iid theta ~ Normal(mu, tau) size=J
    for i in eachindex(y)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

model = eight_schools(y, sigma, J);
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

