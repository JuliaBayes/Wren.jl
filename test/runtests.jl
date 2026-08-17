module WrenTests

using Wren
using Test
using Distributions
using Chairmarks
using VarNames

function run_model(model, latent_source=FromPrior())
    accs = Wren.default_accumulators()
    result = evaluate(model, latent_source, accs)
    display(result)
    display(@b evaluate($model, $latent_source, $accs))
    return result
end

@testset "macro" begin
    @model function f()
        x ~ Normal()
    end
    model = f()
    run_model(model)
    # @info Wren.view_evaluation_fn(model)
end

@testset "arg" begin
    @model function g(m)
        x ~ Normal(m[1])
    end
    d = [100.0]
    run_model(g(d))
end

@testset "cond" begin
    @model function h()
        x ~ Normal()
        y ~ Normal(x)
    end
    run_model(h())
    run_model(h() | VarNamedTuple(y = 1.0))
end

@testset "observe macro" begin
    @model function k()
        x ~ Normal()
        y = x + 1
        @observe y ~ Normal()
    end
    run_model(k())
end

@testset "control flow" begin
    @model function control_flow(n, events)
        if n == 0
            return push!(events, :empty)
        elseif iseven(n)
            push!(events, :even)
        else
            push!(events, :odd)
        end

        xs = Vector{Float64}(undef, n)
        for i in 1:n
            xs[i] ~ Normal()
        end

        i = 1
        while i <= n
            @observe xs[i] ~ Normal()
            i += 1
        end

        push!(events, :finished)
        return
        push!(events, :unreachable)
    end

    values = [0.25, -0.5]
    source = FromParams(VarNamedTuple(; xs=values), nothing)
    events = Symbol[]
    accs = evaluate(control_flow(2, events), source, Wren.default_accumulators())
    expected_logp = sum(logpdf(Normal(), value) for value in values)

    @test events == [:even, :finished]
    raw_xs = get_raw_values(accs).data.xs
    @test VarNames.densify!!(raw_xs) == values
    @test getlogprior(accs) ≈ expected_logp
    @test getloglikelihood(accs) ≈ expected_logp

    events = Symbol[]
    accs = evaluate(control_flow(0, events), FromPrior(), Wren.default_accumulators())
    @test events == [:empty]
    @test get_raw_values(accs) == VarNamedTuple()
end

end # module WrenTests
