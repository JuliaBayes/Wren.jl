# Inference card of the Wren API cheatsheet (../wren-api.typ).

include("conditioning.jl")

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
