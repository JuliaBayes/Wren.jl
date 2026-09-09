# Plaice interface for multivariate distributions.

# For all multivariate distributions, from_vec and to_vec are just identity.
Plaice.from_vec(::MultivariateDist) = Plaice.TypedIdentity()
Plaice.to_vec(::MultivariateDist) = Plaice.TypedIdentity()

# MvNormal: already unconstrained
Plaice.from_unconstrained_vec(::MvNormal) = Plaice.TypedIdentity()
Plaice.to_unconstrained_vec(::MvNormal) = Plaice.TypedIdentity()
Plaice.vec_length(d::MvNormal) = length(d.μ)
Plaice.unconstrained_vec_length(d::MvNormal) = length(d.μ)
function Plaice.optic_vec(d::MvNormal)
    return [VarNames.@opticof(_[i]) for i in eachindex(d.μ)]
end
Plaice.unconstrained_optic_vec(d::MvNormal) = Plaice.optic_vec(d)

# Discrete multivariate distributions: can't transform the support
Plaice.from_unconstrained_vec(::DiscreteMultivariateDist) = Plaice.TypedIdentity()
Plaice.to_unconstrained_vec(::DiscreteMultivariateDist) = Plaice.TypedIdentity()
Plaice.unconstrained_vec_length(d::DiscreteMultivariateDist) = Plaice.vec_length(d)
Plaice.unconstrained_optic_vec(d::DiscreteMultivariateDist) = Plaice.optic_vec(d)

# Multinomial
Plaice.vec_length(d::Multinomial) = length(d.p)
function Plaice.optic_vec(d::Multinomial)
    return [VarNames.@opticof(_[i]) for i in eachindex(d.p)]
end

# Dirichlet: simplex distribution
Plaice.from_unconstrained_vec(::Dirichlet) = Plaice.inverse(Plaice.SimplexBijector())
Plaice.to_unconstrained_vec(::Dirichlet) = Plaice.SimplexBijector()
Plaice.vec_length(d::Dirichlet) = length(d.α)
Plaice.unconstrained_vec_length(d::Dirichlet) = length(d.α) - 1
function Plaice.optic_vec(d::Dirichlet)
    return [VarNames.@opticof(_[i]) for i in eachindex(d.α)]
end
function Plaice.unconstrained_optic_vec(d::Dirichlet)
    return fill(nothing, Plaice.unconstrained_vec_length(d))
end
