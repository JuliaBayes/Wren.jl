module WrenIssue155Tests

using Distributions
import ForwardDiff
import Enzyme
using Test
using VarNames: VarNames
using WrenCore

@testset "indexed sampling sites" begin
    @wren function indexed_sites(index, dist)
        switch ~ Normal()
        x = zeros(2)
        i = ifelse(switch > 0, 1, 2)
        x[index(i)] ~ dist(1, 1)
        x[index(3-i)] ~ dist(-1, 1)
        y = 0.0
        @observe y ~ Normal(x[1], 1)
    end
    @wren function indexed_parent(sub)
        a = Vector{Any}(undef, 1)
        a[1] ~ sub
    end

    @testset "$label" for (label, model, wrap) in (
        ("top level", indexed_sites(identity, Normal), identity),
        (
            "indexed submodel",
            indexed_parent(indexed_sites(identity, Normal)),
            p -> VarNamedTuple(; a=[p]),
        ),
        ("vector indices", indexed_sites(i -> [i], (m, s) -> MvNormal([m], s)), identity),
    )
        initial = wrap(VarNamedTuple(; switch=1.0, x=[1.0, 2.0]))
        vm = VecModel(model, AutoForwardDiff(); latent_source=FromValues(initial))
        @testset "switch=$switch" for switch in (1.0, -1.0)
            p = [switch, 1.0, 2.0]
            named = wrap(VarNamedTuple(; switch, x=[1.0, 2.0]))
            means = switch > 0 ? [1.0, -1.0] : [-1.0, 1.0]
            expected_lp =
                logdensityof(Normal(), switch) +
                logdensityof(Normal(means[1], 1), p[2]) +
                logdensityof(Normal(means[2], 1), p[3]) +
                logdensityof(Normal(p[2], 1), 0.0)
            expected_grad = [-switch, means[1] - 2p[2], means[2] - p[3]]

            @test lp(vm, p) ≈ expected_lp
            value, grad = lp_and_grad(vm, p)
            @test value ≈ expected_lp
            @test grad ≈ expected_grad
            grad_buffer = similar(p)
            value, _ = lp_and_grad!(grad_buffer, vm, p)
            @test value ≈ expected_lp
            @test grad_buffer ≈ expected_grad
            draw = VarNames.densify!!(parameters(vm, p))
            child = label == "indexed submodel" ? draw[@vn(a[1])] : draw
            @test child == VarNamedTuple(; switch, x=[1.0, 2.0])
            @test vector_parameters(vm, named) == p
        end
    end

    @testset "mutable blocks in indexed submodel" for wrap_indices in
                                                      (identity, x -> view(x, :))
        @wren function indexed_blocks(first_indices, second_indices)
            x = zeros(6)
            x[first_indices] ~ MvNormal(ones(3), I)
            x[second_indices] ~ MvNormal(-ones(3), I)
            y = 0.0
            @observe y ~ Normal(x[2], 1)
        end
        model =
            indexed_parent(indexed_blocks(wrap_indices([1, 2, 3]), wrap_indices([4, 5, 6])))
        initial = VarNamedTuple(; a=[VarNamedTuple(; x=collect(1.0:6.0))])
        vm = VecModel(model, AutoForwardDiff(); latent_source=FromValues(initial))
        p = vector_parameters(vm, initial)
        for (a, b) in (
            ([1, 2, 3], [4, 5, 6]),
            ([1, 3, 2], [4, 6, 5]),
            ([2, 1, 3], [5, 4, 6]),
            ([4, 5, 6], [1, 2, 3]),
        )
            model.args.sub.args.first_indices .= a
            model.args.sub.args.second_indices .= b
            ra, rb = first(a) <= 3 ? (1:3, 4:6) : (4:6, 1:3)
            positions = zeros(Int, 6)
            positions[a] = ra
            positions[b] = rb
            x = p[positions]
            expected =
                logdensityof(MvNormal(ones(3), I), p[ra]) +
                logdensityof(MvNormal(-ones(3), I), p[rb]) +
                logdensityof(Normal(), x[2])
            gradient = zeros(6)
            gradient[ra] = 1 .- p[ra]
            gradient[rb] = -1 .- p[rb]
            gradient[positions[2]] -= x[2]
            @test lp(vm, p) ≈ expected
            value, grad = lp_and_grad(vm, p)
            @test value ≈ expected
            @test grad ≈ gradient
            named = VarNamedTuple(; a=[VarNamedTuple(; x)])
            @test VarNames.densify!!(parameters(vm, p))[@vn(a[1])] == named[@vn(a[1])]
            @test vector_parameters(vm, named) == p
        end
    end
end

end # module WrenIssue155Tests
