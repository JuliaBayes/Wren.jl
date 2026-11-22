module WrenEnzymeConditioningTests

using Enzyme, Test, WrenCore, WrenDists

@wren function trial(previous, theta)
    a ~ Normal(theta + previous.a)
    b ~ Normal(a + previous.b)
    return (; a, b)
end

@wren function sequence(data)
    theta ~ Normal()
    previous = (a=0.0, b=0.0)
    trials = Vector{Any}(undef, length(data))
    for i in eachindex(data)
        child = trial(previous, theta)
        if !ismissing(data[i].a)
            child = child | (; a=data[i].a)
        end
        if !ismissing(data[i].b)
            child = child | (; b=data[i].b)
        end
        previous = trials[i] ~ child
    end
end

@testset "Enzyme with sequential internal conditioning" begin
    @static if VERSION > v"1.13.0"
        Record = @NamedTuple{a::Union{Missing,Float64}, b::Union{Missing,Float64}}
        data = Record[(a=1.0, b=missing), (a=missing, b=2.0)]
        values = @vnt begin
            theta := 0.3
            trials[1].b := -0.2
            trials[2].a := 0.4
        end
        expected =
            logdensityof(Normal(), 0.3) +
            logdensityof(Normal(0.3), 1.0) +
            logdensityof(Normal(1.0), -0.2) +
            logdensityof(Normal(1.3), 0.4) +
            logdensityof(Normal(0.2), 2.0)
        for data in (data, view(data, :)), reduce_copies in (false, true)
            model = sequence(data)
            vm = VecModel(
                model,
                AutoEnzyme(; mode=Reverse);
                latent_source=FromValues(values),
                reduce_copies,
            )
            value, gradient = lp_and_grad(vm, [0.3, -0.2, 0.4])
            @test value ≈ expected
            @test gradient ≈ [-0.5, 3.0, 2.7]
        end
    else
        @info "Skipping Enzyme sequential conditioning test on Julia $(VERSION), please see https://github.com/JuliaBayes/Wren.jl/issues/304"
        @test_broken false
    end
end

@wren function indexed_observation(y)
    theta ~ Normal()
    y[1] ~ Normal(theta)
end

@wren function ref_observation(y)
    theta ~ Normal()
    y[] ~ Normal(theta)
end

struct IndexOnly{T}
    data::T
end
Base.getindex(x::IndexOnly, i::Int) = x.data[i]

@testset "Enzyme with indexed observations" begin
    models = (
        indexed_observation((0.0,)) | (y=(1.0,),),
        ref_observation(Ref(0.0)) | (y=Ref(1.0),),
        indexed_observation(IndexOnly((0.0,))) | (y=IndexOnly((1.0,)),),
    )
    for model in models
        vm = VecModel(
            model,
            AutoEnzyme(; mode=Reverse);
            latent_source=FromValues(VarNamedTuple(theta=0.3)),
            reduce_copies=false,
        )
        value, gradient = lp_and_grad(vm, [0.3])
        @test value ≈ logdensityof(Normal(), 0.3) + logdensityof(Normal(0.3), 1.0)
        @test gradient ≈ [0.4]
    end
end

end
