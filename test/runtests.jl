module WrenTests

using Wren
using Test
using Distributions
using Chairmarks
using VarNames

@testset "macro" begin
    @model function f()
        x ~ Normal()
    end
    model = f()
    accs = Wren.default_accumulators()
    display(evaluate(model, accs))
    display(@b evaluate($model, $accs))
    # @info Wren.view_evaluation_fn(model)
end

@testset "arg" begin
    @model function g(m)
        x ~ Normal(m[1])
    end
    d = [100.0]
    model = g(d)
    accs = Wren.default_accumulators()
    display(evaluate(model, accs))
    display(@b evaluate($model, $accs))
end

@testset "cond" begin
    @model function h()
        x ~ Normal()
        y ~ Normal(x)
    end
    model = h()
    accs = Wren.default_accumulators()
    display(evaluate(model, accs))
    display(@b evaluate($model, $accs))

    model = h() | VarNamedTuple(y = 1.0)
    accs = Wren.default_accumulators()
    display(evaluate(model, accs))
    display(@b evaluate($model, $accs))
end

@testset "observe macro" begin
    @model function k()
        x ~ Normal()
        y = x + 1
        @observe y ~ Normal()
    end
    model = k()
    accs = Wren.default_accumulators()
    display(evaluate(model, accs))
    display(@b evaluate($model, $accs))
end

end # module WrenTests
