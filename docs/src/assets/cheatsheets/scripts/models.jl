# Models card of the Wren API cheatsheet (../wren-api.typ).
# Each `#region name` … `#endregion` block is one code box on the card; code outside the
# regions only makes the file run on its own, which Wren/test/cheatsheets.jl checks.

using Wren, Wren.Dists

y = [0, 1, 1, 0, 1, 1, 1, 0]

@wren function sample_statements(y, n)
    #region sample-statements
    # Latent unless conditioned
    x ~ Normal(0, 1)

    # Always observed
    @observe y ~ Normal(x, 1)

    # Prior of length n
    @iid mu ~ Normal() size=n

    # Explicit sample statement with a site name
    @sample x ~ Normal() name=:prior
    #endregion
end

simulate(sample_statements(0.5, 3))

#region model
# Data and shapes arrive as arguments
@wren function coin(y)
    p ~ Beta(1, 1)
    # Vectorise independent ~ statements
    @batch for i in eachindex(y)
        y[i] ~ Bernoulli(p)
    end
    # Generated quantity, stored with every draw
    @gq odds = p / (1 - p)
end

# Instantiate
model = coin(y)
#endregion
