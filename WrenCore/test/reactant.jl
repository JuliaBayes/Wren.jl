module WrenReactantTests

@info "testing reactant.jl"

using ADTypes: AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using Test
using WrenCore
using WrenDists
using LinearAlgebra

@wren function reactant_model(a)
    x ~ Normal(a, 1.0)
end
@wren function reactant_submodel(inner)
    x ~ inner
end

@testset "Reactant" begin
    adtype = AutoReactant(AutoEnzyme(; mode=Reverse))
    vecmodel = VecModel(reactant_model(1.0), adtype; reduce_copies=true)
    params = rand(vecmodel)

    @test lp(vecmodel, params) isa Float64
    logdensity, grad = lp_and_grad(vecmodel, params)
    @test logdensity isa Float64
    @test grad isa Vector{Float64}

    nested = VecModel(reactant_submodel(reactant_model(1.0)), adtype; reduce_copies=true)
    nested_logdensity, nested_grad = lp_and_grad(nested, params)
    @test nested_logdensity ≈ logdensity
    @test nested_grad ≈ grad
end

@wren function branched_latent()
    s ~ Normal()
    if s > 0
        x ~ Normal(1.0, 2.0)
    else
        x ~ Normal(-1.0, 1.0)
    end
end

@wren function branched_observation(y)
    s ~ Normal()
    # Generated names must not collide with user locals.
    var"#offset" = 0.25
    var"_wren_#offset" = 0.75
    if s > 0
        @observe y ~ Normal(s + var"#offset" + var"_wren_#offset", 2.0)
    else
        @observe y ~ Normal(s + var"#offset" + var"_wren_#offset", 1.0)
    end
end

@wren function branched_alias()
    s ~ Normal()
    if s > 0
        y ~ Normal()
    else
        z ~ Normal()
    end
end

@wren function indexed_buffer(A, indices, y)
    x ~ MvNormalScale(zeros(size(A, 2)), I)
    buffer = similar(y)
    mul!(buffer, A, x)
    @observe y[indices] ~ MvNormalScale(buffer[indices], I)
end

@testset "Reactant branches and host indices" begin
    adtype = AutoReactant(AutoEnzyme(; mode=Reverse))
    initial = FromValues(VarNamedTuple(; s=1.0, x=0.5))
    latent = VecModel(branched_latent(), adtype; latent_source=initial)
    observation = VecModel(branched_observation(0.5), adtype; latent_source=initial)
    for s in (1.0, -1.0)
        scale = s > 0 ? 2.0 : 1.0
        mean = s > 0 ? 1.0 : -1.0
        for (vm, p, expected, gradient) in (
            (
                latent,
                [s, 0.5],
                logdensityof(Normal(), s) + logdensityof(Normal(mean, scale), 0.5),
                [-s, (mean - 0.5) / scale^2],
            ),
            (
                observation,
                [s],
                logdensityof(Normal(), s) + logdensityof(Normal(s + 1.0, scale), 0.5),
                [-s + (-0.5 - s) / scale^2],
            ),
        )
            @test lp(vm, p) ≈ expected
            value, grad = lp_and_grad(vm, p)
            @test value ≈ expected
            @test grad ≈ gradient
        end
    end
    for aliases in (nothing, ((@vn(y), @vn(z)),))
        vm = VecModel(
            branched_alias(),
            adtype;
            aliases,
            latent_source=FromValues(VarNamedTuple(; s=1.0, y=0.5)),
        )
        for s in (1.0, -1.0)
            p = [s, 0.5]
            value, grad = lp_and_grad(vm, p)
            @test value ≈ -log(2pi) - sum(abs2, p) / 2
            @test grad ≈ -p
        end
    end
    A, indices, y = [1.0 2.0; 3.0 4.0], [2, 1], [0.5, 1.0]
    p = [0.2, -0.1]
    vm = VecModel(
        indexed_buffer(A, indices, y),
        adtype;
        latent_source=FromValues(VarNamedTuple(; x=p)),
    )
    expected = -2log(2pi) - (sum(abs2, p) + sum(abs2, y - A * p)) / 2
    value, grad = lp_and_grad(vm, p)
    @test value ≈ expected
    @test grad ≈ -p + A' * (y - A * p)
    @test indices == [2, 1]
end

end # module WrenReactantTests
