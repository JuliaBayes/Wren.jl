# Conditioning & evaluation card of the Wren API cheatsheet (../wren-api.typ).

include("models.jl")

y_obs = y

#region conditioning
# Condition on observed y
cond = model | (; y=y_obs)

# Make y latent again
decondition(cond, :y)
#endregion
