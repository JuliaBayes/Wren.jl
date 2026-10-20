module Issue55

using WrenCore, WrenDists, Enzyme, Test

@wren function inner()
    x ~ Normal()
end
@wren function outer(a)
    for i in eachindex(a)
        a[i] ~ inner()
    end
end

@testset "#55: type stability with UncollectedParams" begin
    model = outer(zeros(3))

    @test VecModel(model, WrenCore.DEFAULT_AD) isa Any
end

end
