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
    display(model.evaluation_fn(accs, model.args))
    display(@b model.evaluation_fn($accs, $(model.args)))
    @info Wren.view_evaluation_fn(model)
end

@testset "arg" begin
    @model function g(m)
        x ~ Normal(m[1])
    end
    d = [100.0]
    model = g(d)
    accs = Wren.default_accumulators()
    display(model.evaluation_fn(accs, model.args))
    display(@b model.evaluation_fn($accs, $(model.args)))
    @info Wren.view_evaluation_fn(model)
end

end # module WrenTests
