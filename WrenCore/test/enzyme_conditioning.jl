module WrenEnzymeConditioningTests

using Enzyme, Random, Test, WrenCore, WrenDists

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
end

# A mutable pointer-free record must still be deepcopied, preserving aliases between
# elements while separating the model's state from the caller's state.
mutable struct MutableRecord
    value::Float64
end

@wren function mutate_shared(data)
    data[1].value += 1.0
    return data[2].value
end

@wren function mutate_nested(data)
    data[1].value[1] += 1.0
    return data[2].value[1]
end

@testset "Deep copies retain mutable element aliases" begin
    for as_view in (false, true), nested in (false, true)
        shared = nested ? (value=[0.0],) : MutableRecord(0.0)
        data = [shared, shared]
        data = as_view ? view(data, :) : data
        model = nested ? mutate_nested(data) : mutate_shared(data)
        result, _ = evaluate(
            model,
            FromPrior(Xoshiro(468)),
            OriginalSpace(),
            WrenCore.default_accumulators(),
            EvalContext{false}(),
        )
        @test result == 1.0
        @test (nested ? shared.value[1] : shared.value) == 0.0
    end
end

end
