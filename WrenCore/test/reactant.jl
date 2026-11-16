module WrenReactantTests

@info "testing reactant.jl"

using ADTypes: AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using Test
using WrenCore
using WrenDists
using Plaice

include("reactant_models.jl")
using .ReactantModelFixtures

@wren function reactant_model(a)
    x ~ Normal(a, 1.0)
end
@wren function reactant_submodel(inner)
    x ~ inner
end
@wren function aliased_argument(pair)
    pair[1][1] += 1
    return pair[2][1]
end

function transform_columns(X)
    return mapslices(X; dims=1) do x
        Reactant.@allowscalar vcat(
            Plaice.with_logabsdet_jacobian(Plaice.Truncate(x[1], x[2]), x[3])...,
        )
    end
end

@testset "Reactant" begin
    adtype = AutoReactant(AutoEnzyme(; mode=Reverse))
    vecmodel = VecModel(reactant_model(1.0), adtype; reduce_copies=true)
    params = vector_parameters(vecmodel)

    @test lp(vecmodel, params) isa Float64
    logdensity, grad = lp_and_grad(vecmodel, params)
    @test logdensity isa Float64
    @test grad isa Vector{Float64}

    nested = VecModel(reactant_submodel(reactant_model(1.0)), adtype; reduce_copies=true)
    nested_logdensity, nested_grad = lp_and_grad(nested, params)
    @test nested_logdensity ≈ logdensity
    @test nested_grad ≈ grad

    initial = FromValues(VarNamedTuple(x=1.0))
    y = [0.2, 0.4]
    branches = VecModel(branch_model(y), adtype; latent_source=initial)
    native_branches =
        VecModel(branch_model(y), AutoEnzyme(; mode=Reverse); latent_source=initial)
    for x in (-0.5, 0.5)
        mu = x > 0 ? shift(x) : 2x - 1
        expected = logdensityof(Normal(), x) + logdensityof(Normal(mu, 2.0), last(y))
        expected_gradient = -x + (last(y) - mu) * (x > 0 ? 1 : 2) / 4
        if x > 0
            expected += sum(logdensityof.(Ref(Normal(mu)), y))
            expected_gradient += sum(y .- mu)
        end
        value, gradient = lp_and_grad(branches, [x])
        @test value ≈ expected
        @test gradient ≈ [expected_gradient]
        native_value, native_gradient = lp_and_grad(native_branches, [x])
        @test native_value ≈ expected
        @test native_gradient ≈ [expected_gradient]
    end
    nested_branches = VecModel(
        reactant_submodel(branch_model(y)),
        adtype;
        latent_source=FromValues(@vnt x.x := 1.0),
    )
    value, gradient = lp_and_grad(nested_branches, [0.5])
    expected, expected_gradient = lp_and_grad(branches, [0.5])
    @test value ≈ expected
    @test gradient ≈ expected_gradient

    converted = Reactant.to_rarray(branch_model(y))
    joint(model, x) = Reactant.@allowscalar logjoint(simulate(model, VarNamedTuple(x=x[1])))
    input = Reactant.to_rarray([0.5])
    compiled_joint = Reactant.@compile joint(converted, input)
    for x in (-0.5, 0.5)
        @test Float64(compiled_joint(converted, Reactant.to_rarray([x]))) ≈
              lp(branches, [x])
    end
    shared = [0.0]
    alias_model = Reactant.to_rarray(aliased_argument((shared, shared)))
    @test Float64(return_value(simulate(alias_model))) == 1.0
    @test shared == [0.0]

    bounded = VecModel(
        bounded_model(3),
        adtype;
        latent_source=FromValues(VarNamedTuple(upper=2.0, rate=0.75)),
    )
    x = [0.2, 0.5]
    upper, fraction = exp(x[1]), 1 / (1 + exp(-x[2]))
    rate = upper * fraction
    value, gradient = lp_and_grad(bounded, x)
    @test value ≈
          -upper + x[1] + log(fraction) + log1p(-fraction) + 3log(rate) - rate - log(6.0)
    @test gradient ≈ [4 - upper - rate, 1 - 2fraction + (3 - rate) * (1 - fraction)]

    bounds = [0.0 0.0 -Inf -Inf; 2.0 Inf 3.0 Inf; 0.5 0.5 0.5 0.5]
    expected_transforms = mapreduce(hcat, eachcol(bounds)) do x
        vcat(Plaice.with_logabsdet_jacobian(Plaice.Truncate(x[1], x[2]), x[3])...)
    end
    bounds_input = Reactant.to_rarray(bounds)
    @test Array(Reactant.@jit transform_columns(bounds_input)) ≈ expected_transforms

    array_model = VecModel(
        array_branch(0.2),
        adtype;
        latent_source=FromValues(VarNamedTuple(beta=[1.0, 0.5])),
    )
    @test last(lp_and_grad(array_model, [0.5, 0.2])) ≈ [-1.0, -0.7]

    # Preparation adds the initial rate once. Evaluation must reuse that seed.
    expected_mutation = [3log(4.0) - 4 - log(6.0), 0.5 * (3 / 4 - 1)]
    mutating = VecModel(
        mutating_model([2.0], 3),
        adtype;
        lp_function=loglikelihood,
        latent_source=FromValues(VarNamedTuple(rate=1.5)),
    )
    for _ in 1:2
        @test vcat(lp_and_grad(mutating, [log(0.5)])...) ≈ expected_mutation
    end

    conditioned_mutating = VecModel(
        condition(mutating_model([2.0], 3), (; buffer=[2.0])),
        adtype;
        lp_function=loglikelihood,
        latent_source=FromValues(VarNamedTuple(rate=1.5)),
    )
    for _ in 1:2
        @test vcat(lp_and_grad(conditioned_mutating, [log(0.5)])...) ≈ expected_mutation
    end

    early = VecModel(control_model(0.2, true), adtype; latent_source=initial)
    @test lp(early, [2.0]) ≈ logdensityof(Normal(), 2.0)
    terminal = VecModel(control_model(0.2, false), adtype; latent_source=initial)
    @test last(lp_and_grad(terminal, [2.0])) ≈ [-3.8]
end

end # module WrenReactantTests
