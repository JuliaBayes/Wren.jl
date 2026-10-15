module Issue132

using WrenCore, WrenDists, Enzyme, Test

@testset "#132 Enzyme failure with deepcopied Int" begin
    @wren function inner(n)
        eps ~ MvNormalScale(zeros(n - 1), I)
        return cumsum(eps)
    end
    @wren function outer(inner, cases)
        Z = (rw ~ inner)
        s = sum(Z)
        for t in eachindex(cases)
            cases[t] ~ Poisson(exp(s) * 10)
        end
    end
    model = outer(inner(11), zeros(Int, 10)) | VarNamedTuple(; cases=rand(1:20, 10))
    @test VecModel(model, WrenCore.DEFAULT_AD) isa Any
end

end
