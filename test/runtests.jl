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

macro sample_x()
    return esc(:(x ~ Normal()))
end

macro observe_y()
    return esc(:(@observe y ~ Normal()))
end

@testset "macro" begin
    @model function f()
        x ~ Normal()
    end
    model = f()
    run_model(model)
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
    run_model(h() | VarNamedTuple(y=1.0))
end

@testset "observe macro" begin
    @model function k()
        x ~ Normal()
        y = x + 1
        @observe y ~ Normal()
    end
    run_model(k())
end

@testset "macro expansion" begin
    @model function expanded_macros()
        @sample_x
        y = 0.25
        @observe_y
        xs = Vector{Float64}(undef, 2)
        @inbounds for i in eachindex(xs)
            xs[i] ~ Normal()
        end
    end

    values = (0.5, 1.0, 2.0)
    params = VarNamedTuple(; x=values[1], xs=collect(values[2:3]))
    source = FromParams(params, nothing)
    accs = evaluate(expanded_macros(), source, Wren.default_accumulators())

    @test getlogprior(accs) ≈ sum(logpdf(Normal(), value) for value in values)
    @test getloglikelihood(accs) ≈ logpdf(Normal(), 0.25)
end

@testset "nested expressions" begin
    @model function nested_if(flag)
        x = if flag
            y ~ Normal()
            y
        else
            z ~ Normal()
            z
        end
        @observe x ~ Normal()
    end

    params = VarNamedTuple(; y=0.5, z=2.0)
    source = FromParams(params, nothing)

    accs = evaluate(nested_if(true), source, Wren.default_accumulators())
    @test getlogprior(accs) ≈ logpdf(Normal(), 0.5)
    @test getloglikelihood(accs) ≈ logpdf(Normal(), 0.5)

    accs = evaluate(nested_if(false), source, Wren.default_accumulators())
    @test getlogprior(accs) ≈ logpdf(Normal(), 2.0)
    @test getloglikelihood(accs) ≈ logpdf(Normal(), 2.0)
end

@testset "tilde in value position" begin
    @model function parser_inner()
        return 1
    end

    @model function tilde_value_position()
        retval = x ~ parser_inner()
    end

    stmt = only(filter(x -> x isa Wren.ModelExpr, tilde_value_position().stmts))
    @test stmt isa Wren.JuliaExpr
    @test stmt.head == :(=)
    @test stmt.args[1] == :retval
    @test stmt.args[2] isa Wren.Sample
    @test stmt.args[2].lhs == :x
    @test stmt.args[2].rhs isa Wren.JuliaExpr
    @test stmt.args[2].rhs.head == :call
    @test stmt.args[2].rhs.args == [:parser_inner]
end

@testset "distribution tilde value" begin
    @model function distribution_tilde_value()
        retval = x ~ Normal()
        @observe retval ~ Normal()
    end

    x = 0.5
    source = FromParams(VarNamedTuple(; x), nothing)
    accs = evaluate(distribution_tilde_value(), source, Wren.default_accumulators())
    @test get_raw_values(accs) == VarNamedTuple(; x)
    @test getlogprior(accs) ≈ logpdf(Normal(), x)
    @test getloglikelihood(accs) ≈ logpdf(Normal(), x)

    conditioned_model = distribution_tilde_value() | VarNamedTuple(; x)
    accs = evaluate(conditioned_model, FromPrior(), Wren.default_accumulators())
    @test get_raw_values(accs) == VarNamedTuple()
    @test getlogprior(accs) == 0
    @test getloglikelihood(accs) ≈ 2 * logpdf(Normal(), x)
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
