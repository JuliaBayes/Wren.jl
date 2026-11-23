module WrenBranchAliasTests

using Distributions
import ForwardDiff
import Enzyme
using Test
using WrenCore

@testset "branch aliases" begin
    @wren function alias_child()
        y ~ Normal()
    end
    @wren function alias_branches(sub)
        x ~ Normal()
        child ~ sub
        if x > 0
            y ~ Exponential()
            common ~ Normal(3, 2)
        else
            common ~ Normal(3, 2)
            z ~ Beta(2, 3)
        end
    end
    initial = VarNamedTuple(; x=1.0, child=VarNamedTuple(; y=0.5), y=exp(0.4), common=4.0)
    for aliases in (nothing, ((@vn(y), @vn(z)),))
        vm = VecModel(
            alias_branches(alias_child()),
            AutoForwardDiff();
            latent_source=FromValues(initial),
            aliases,
        )
        for x in (1.0, -1.0, 1.0)
            p = [x, 0.5, 0.4, 4.0]
            z = 1 / (1 + exp(-p[3]))
            branch_lp = x > 0 ? -exp(p[3]) + p[3] : log(12) + 2log(z) + 3log1p(-z)
            expected =
                logdensityof(Normal(), x) +
                logdensityof(Normal(), p[2]) +
                logdensityof(Normal(3, 2), p[4]) +
                branch_lp
            gradient = [-x, -p[2], x > 0 ? 1-exp(p[3]) : 2-5z, -0.25]
            @test lp(vm, p) ≈ expected
            value, grad = lp_and_grad(vm, p)
            @test value ≈ expected
            @test grad ≈ gradient
            draw = parameters(vm, p)
            @test draw[@vn(child.y)] == 0.5
            @test draw[@vn(common)] == 4.0
            @test draw[x > 0 ? @vn(y) : @vn(z)] ≈ (x > 0 ? exp(p[3]) : z)
            @test vector_parameters(vm, draw) ≈ p
        end
    end
end

@testset "explicit aliases with copy reduction" begin
    @wren function loop_aliases()
        x ~ Normal()
        for _ in 1:1
            if x > 0
                y ~ Normal()
            else
                z ~ Normal()
            end
        end
    end
    vm = VecModel(
        loop_aliases(),
        AutoEnzyme(; mode=Reverse);
        aliases=((@vn(y), @vn(z)),),
        latent_source=FromValues(VarNamedTuple(; x=0.5, y=0.4)),
    )
    for x in (0.5, -0.5)
        p = [x, 0.4]
        value, gradient = lp_and_grad(vm, p)
        @test value ≈ -log(2pi) - sum(abs2, p)/2
        @test gradient ≈ -p
    end
end

end # module WrenBranchAliasTests
