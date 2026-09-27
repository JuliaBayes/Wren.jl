module WrenSitenameTests

@info "testing sitenames.jl"

using WrenCore
using WrenCore: SitenameCountAccumulator, SeenMultipleTimes, SITENAMECOUNT_ACCNAME, getacc
using Test
using Distributions
using VarNames: getsym

@testset "sitenames.jl" begin
    @testset "SitenameCountAccumulator records sites hit exactly once" begin
        @wren function inner()
            z ~ Normal()
        end
        @wren function sites(N, sub)
            a ~ Normal()
            x = zeros(N)
            for i in 1:N
                x[i] ~ Normal()
            end
            s ~ sub
        end
        accs = AccumulatorTuple(SitenameCountAccumulator())
        _, accs = evaluate(sites(3, inner()), FromPrior(), OriginalSpace(), accs)
        entries = getacc(accs, Val(SITENAMECOUNT_ACCNAME)).entries

        sitenames = collect(keys(entries))
        @test issetequal(
            [entries[k] for k in sitenames],
            [@vn(a), SeenMultipleTimes(), @vn(s.z)],
        )
        # The submodel's sites are prefixed with the submodel's VarName, not its sitename.
        submodel_site = only(k for k in sitenames if entries[k] == @vn(s.z))
        @test getsym(submodel_site) === :s
    end

    @testset "VecModel falls back to VarName lookup when a site is not seen during construction" begin
        # Both branches produce the same VarName with the same vector size, so the model can
        # be vectorised. Only the branch hit during VecModel construction will have its
        # sitename recorded though.
        @wren function branchy(flag)
            if flag[]
                x ~ Exponential(1.0)
            else
                x ~ Gamma(2.0, 3.0)
            end
        end
        vm = VecModel(branchy(Ref(true)), nothing)
        y = 0.3
        @test lp(vm, [y]) ≈ logpdf(Exponential(1.0), exp(y)) + y
        # Model arguments are copied on construction, so this has to go through `vm`.
        vm.model.args.flag[] = false
        @test lp(vm, [y]) ≈ logpdf(Gamma(2.0, 3.0), exp(y)) + y
    end
end

end # module WrenSitenameTests
