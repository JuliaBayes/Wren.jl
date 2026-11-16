module ReactantModelFixtures

using WrenCore, WrenDists
using Reactant: @trace

export branch_model, bounded_model, control_model, mutating_model, array_branch, shift

const offset = 0.3
shift(x) = x + offset
@wren function branch_model(y)
    x ~ Normal()
    @trace mu = if x > 0
        shift(x)
    else
        2x - 1
    end
    @trace for i in eachindex(y)
        @trace if x > 0
            @observe y[i] ~ Normal(mu)
        end
    end
    n = length(y)
    @observe y[n] ~ Normal(mu, 2.0)
end

@wren function bounded_model(count)
    upper ~ Exponential()
    rate ~ Uniform(0.0, upper)
    @observe count ~ Poisson(rate)
end

@wren function control_model(y, flag)
    x ~ Normal()
    if flag
        return x
    end
    @trace if x > 0
        @trace if x > 1
            @observe y ~ Normal(x)
        end
    end
end

@wren function mutating_model(buffer, count)
    rate ~ Exponential()
    buffer[1] += rate
    @observe count ~ Poisson(buffer[1])
end

@wren function array_branch(y)
    beta ~ iid(Normal(), 2)
    @trace if beta[1] > 0
        @observe y ~ Normal(sum(beta))
    else
        @observe y ~ Normal(2beta[1] - beta[2])
    end
end

end
