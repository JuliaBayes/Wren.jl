using WrenCore, Distributions, Test

@testset "#149 typed, default, and keyword model arguments" begin
    @wren function f(a::Real, b=2.0; c=3.0)
        x ~ Normal()
        return (a, b, c)
    end
    for (model, expected) in (
        (f(1.0), (a=1.0, b=2.0, c=3.0)),
        (f(1.0, 4.0; c=5.0), (a=1.0, b=4.0, c=5.0)),
    )
        @test model.args == expected
        retval, _ = evaluate(
            model, FromPrior(), OriginalSpace(), WrenCore.default_accumulators()
        )
        @test retval == values(expected)
    end
end
