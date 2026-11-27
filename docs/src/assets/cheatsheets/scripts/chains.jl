# Chains card of the Wren API cheatsheet (../wren-api.typ).

include("inference.jl")

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
