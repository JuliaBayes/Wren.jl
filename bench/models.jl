module BenchmarkModels

using Distributions
using LinearAlgebra: I
using Wren: Wren, VarNamedTuple
using DynamicPPL: DynamicPPL

Wren.@model function wren_normal()
    x ~ Normal()
    return nothing
end

DynamicPPL.@model function dppl_normal()
    x ~ Normal()
    return nothing
end

const EIGHT_SCHOOLS_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SCHOOLS_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
const EIGHT_SCHOOLS_J = 8

Wren.@model function wren_eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ MvNormal(fill(mu, J), tau^2 * I)
    for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

DynamicPPL.@model function dppl_eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ MvNormal(fill(mu, J), tau^2 * I)
    for i in 1:J
        y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

function models()
    wren_schools = wren_eight_schools(EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, EIGHT_SCHOOLS_J)
    dppl_schools = dppl_eight_schools(EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, EIGHT_SCHOOLS_J)
    return (
        (name="normal", wren=wren_normal(), dppl=dppl_normal()),
        (name="eight schools", wren=wren_schools, dppl=dppl_schools),
    )
end

end # module BenchmarkModels
