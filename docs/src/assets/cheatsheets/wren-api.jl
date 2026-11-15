# Code shown on the Wren API cheatsheet (wren-api.typ).
# Each `#region name` … `#endregion` block is one code box on the sheet; code outside the
# regions only makes the file run top to bottom, which Wren/test/cheatsheets.jl checks.

using Wren, Wren.Dists

y = [0, 1, 1, 0, 1, 1, 1, 0]
y_obs = y

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

#region conditioning
# Condition on observed y
cond = model | (; y=y_obs)

# Make y latent again
decondition(cond, :y)
#endregion

#region sampling
import Enzyme

# Draw from the prior, returns `ModelResult`
res = simulate(model)

# 1000 draws from the prior, returns `VNChain`
prior(model, 1000, 4)

# Posterior draws with NUTS
chain = nuts(cond, 1000, 4; progress=false)

# Simulate y for every posterior draw
pred = simulate(model, chain)
#endregion

#region point-estimates
import Optimization
using OptimizationOptimJL: LBFGS

# Posterior mode; MLE ignores the prior
fit = MAP(cond, LBFGS())

# Estimates and convergence
fit.params, fit.solution.retcode
#endregion

#region indexing
# One chain only
chain[@vn(p), chain=2]

# Sampler statistics
chain[:is_divergent]
#endregion

#region summaries
# Mean of every variable
mn = mean(chain)

# Look up one variable
mn[@vn(p)]

# Per-chain means
mean(chain; dims=:iter)[@vn(p), chain=2]

# All statistics; pick one with stat
summarystats(chain)[@vn(p), stat=:mean]
#endregion

#region diagnostics
# Convergence of one variable
rhat(chain)[@vn(p)]

# Effective sample size of every variable
ess(chain)
#endregion
