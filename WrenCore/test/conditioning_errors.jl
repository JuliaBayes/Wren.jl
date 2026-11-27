module ConditioningErrorTests

using WrenCore, WrenDists, Enzyme, Test

@wren function child(theta=0.0)
    a ~ Normal(theta)
end

@wren function parent(children, inner, indices)
    children[indices...] ~ inner
end

@wren function property_parent(children, inner)
    children.child ~ inner
end

struct IndexOnly{T}
    data::T
end
Base.getindex(x::IndexOnly, i::Int) = x.data[i]

struct InterruptedIndex end
Base.getindex(::InterruptedIndex, ::Int) = throw(InterruptException())

@testset "Conditioning lookup context" begin
    model = parent(nothing, child(), (2,)) | (children=[(a=1.0,)],)
    @test_throws r"(?s)conditioned submodel namespace children\[2\].*a\s*=\s*1\.0.*BoundsError" logjoint(
        model,
        VarNamedTuple(),
    )

    for (supplied, indices) in ((Vector{Any}(undef, 1), (1,)), (Ref{NamedTuple}(), ()))
        model = parent(nothing, child(), indices) | (children=supplied,)
        @test_throws r"(?s)conditioned submodel namespace children.*UndefRefError" logjoint(
            model,
            VarNamedTuple(),
        )
    end

    model = parent(nothing, child(), (1,)) | (children=[1.0],)
    @test_throws r"(?s)conditioned submodel namespace children\[1\].*NamedTuple" logjoint(
        model,
        VarNamedTuple(),
    )

    for (supplied, indices) in
        ((((a=1.0,),), (1,)), (Ref((a=1.0,)), ()), (IndexOnly([(a=1.0,)]), (1,)))
        model = parent(nothing, child(), indices) | (children=supplied,)
        @test logjoint(model, VarNamedTuple()) ≈ logdensityof(Normal(), 1.0)
    end
    for supplied in ((), IndexOnly([(a=1.0,)]))
        model = parent(nothing, child(), (2,)) | (children=supplied,)
        @test_throws r"(?s)conditioned submodel namespace children\[2\].*BoundsError" logjoint(
            model,
            VarNamedTuple(),
        )
    end

    model = property_parent(nothing, child() | (a=1.0,)) | (children=(tag=0.0,),)
    @test logjoint(model, VarNamedTuple()) ≈ logdensityof(Normal(), 1.0)
    model = parent(nothing, child(), (1,)) | (children=InterruptedIndex(),)
    @test_throws InterruptException logjoint(model, VarNamedTuple())
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
