module Issue146

using WrenCore
using WrenDists
using ForwardDiff
using Random
using Test

@testset "#146 per-thread model copies own their arguments" begin
    # The model writes its draws into `x` and then reads the whole array back. A single
    # element can stay in a register between the write and the read, which hides the race.
    @wren function writes_arg(x, y)
        n = length(y)
        x[1:n] ~ iid(Normal(0.0, 1.0), (n,))
        @observe y ~ WrenDists.BatchedNormal(x, fill(0.01, n))
    end
    @wren function wraps_arg(sub)
        a ~ sub
    end

    n = 1000
    vm = VecModel(writes_arg(zeros(n), zeros(n)), AutoForwardDiff())

    @testset "copies are independent" begin
        copied = WrenCore._copy_for_thread(vm)
        @test copied.model.args.x !== vm.model.args.x
        @test copied.lp_func.model_args.x !== vm.lp_func.model_args.x
        @test copied.model.evaluation_fn === vm.model.evaluation_fn

        outer = wraps_arg(writes_arg(zeros(n), zeros(n)))
        @test WrenCore._copy_for_thread(outer).args.sub.args.x !== outer.args.sub.args.x
    end

    @testset "concurrent evaluation matches serial" begin
        params = [fill(0.001 * t, n) for t in 1:4]
        serial = [lp(vm, p) for p in params]
        tasks = map(1:4) do t
            Threads.@spawn begin
                copied = WrenCore._copy_for_thread(vm)
                all(lp(copied, params[t]) == serial[t] for _ in 1:500)
            end
        end
        @test all(fetch, tasks)

        model = vm.model
        draws = fetch.([
            Threads.@spawn begin
                copied = WrenCore._copy_for_thread(model)
                [Draw(FromPrior(Xoshiro(100t + i)), copied) for i in 1:50]
            end for t in 1:4
        ])
        @test all(
            d -> logjoint(d) ≈ logjoint(model, parameters(d)), reduce(vcat, draws)
        )
    end
end

end # module Issue146
