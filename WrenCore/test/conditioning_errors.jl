module ConditioningErrorTests

using WrenCore, WrenDists, Enzyme, Test

@wren function child(theta=0.0)
    a ~ Normal(theta)
    return a
end

@wren function parent(children, inner, n)
    result = 0.0
    for i in 1:n
        result += children[i] ~ inner
    end
    return result
end

@wren function property_parent(children, inner)
    return children.child ~ inner
end

@wren function ref_parent(child, inner)
    return child[] ~ inner
end

@wren function indexed_value(y, i)
    return y[i] ~ Normal()
end

@wren function last_value(y)
    return y[end] ~ Normal()
end

@wren function keyword_parent(children, inner)
    return children[row=1] ~ inner
end

struct IndexOnly{T}
    value::T
end
function Base.getindex(x::IndexOnly, i::Int)
    if i == 1
        return x.value
    else
        throw(BoundsError(x, i))
    end
end
Base.getindex(x::IndexOnly; row) = x[row]

struct InterruptedIndex end
Base.getindex(::InterruptedIndex, ::Int) = throw(InterruptException())

result(model) = first(
    evaluate(
        model,
        FromValues(VarNamedTuple()),
        OriginalSpace(),
        AccumulatorTuple(),
        EvalContext{false}(),
    ),
)

@testset "Conditioning lookup context" begin
    model = parent(Vector{Any}(undef, 2), child(), 2) | (children=[(a=1.0,)],)
    @test_throws r"(?s)conditioned submodel namespace children\[2\].*a\s*=\s*1\.0" result(
        model,
    )

    model = parent(Vector{Any}(undef, 1), child(), 1) | (children=[1.0],)
    @test_throws r"(?s)conditioned submodel namespace children\[1\].*NamedTuple" result(
        model,
    )

    model = parent(Vector{Any}(undef, 1), child(), 1) | (children=Vector{Any}(undef, 1),)
    @test_throws r"conditioned submodel namespace children\[1\]" result(model)

    for supplied in (((a=1.0,),), IndexOnly((a=1.0,)))
        model = parent(Vector{Any}(undef, 1), child(), 1) | (children=supplied,)
        @test result(model) == 1.0
    end
    model = parent(Vector{Any}(undef, 2), child(), 2) | (children=IndexOnly((a=1.0,)),)
    @test_throws r"conditioned submodel namespace children\[2\]" result(model)

    model = ref_parent(Ref(nothing), child()) | (child=Ref((a=1.0,)),)
    @test result(model) == 1.0
    @test result(indexed_value((0.0,), 1) | (y=(1.0,),)) == 1.0
    @test_throws r"conditioned variable y\[1\]" result(indexed_value((0.0,), 1) | (y=(),))
    @test result(last_value([0.0]) | (y=[1.0],)) == 1.0
    @test result(indexed_value([0.0], CartesianIndex(1)) | (y=[1.0],)) == 1.0
    @test result(indexed_value([0.0], Int128(1)) | (y=[1.0],)) == 1.0
    @test_throws r"conditioned variable y\[true\]" result(
        indexed_value([0.0], true) | (y=[1.0],),
    )
    model = ref_parent(Ref(nothing), child()) | (child=Ref{NamedTuple}(),)
    @test_throws r"conditioned submodel namespace child\[\]" result(model)
    model = keyword_parent(Vector{Any}(undef, 1), child()) | (children=IndexOnly((a=1.0,)),)
    @test result(model) == 1.0

    model = property_parent((child=nothing,), child() | (a=1.0,)) | (children=(tag=0.0,),)
    @test result(model) == 1.0
    model = parent(Vector{Any}(undef, 1), child(), 1) | (children=InterruptedIndex(),)
    @test_throws InterruptException result(model)
end

@wren function gradient_parent(children)
    theta ~ Normal()
    for i in eachindex(children)
        children[i] ~ child(theta)
    end
end

@testset "Conditioned submodel gradients" begin
    data = [(a=1.0,), (a=2.0,)]
    model = gradient_parent(Vector{Any}(undef, 2)) | (children=data,)
    vm = VecModel(
        model,
        AutoEnzyme(; mode=Reverse);
        latent_source=FromValues(VarNamedTuple(theta=0.3)),
        reduce_copies=false,
    )
    value, gradient = lp_and_grad(vm, [0.3])
    @test value ≈
          logdensityof(Normal(), 0.3) +
          sum(logdensityof(Normal(0.3), y) for y in (1.0, 2.0))
    @test gradient ≈ [2.1]
    @test data == [(a=1.0,), (a=2.0,)]
end

end
