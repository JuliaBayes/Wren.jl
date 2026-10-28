module WrenAliasingTests

@info "testing aliasing.jl"

using WrenCore
using Distributions
using ForwardDiff, ReverseDiff, Mooncake, Enzyme
using Chairmarks
using Test


@testset "aliasing.jl" begin
    @testset "type stability of _dealias" begin
        @inferred WrenCore._dealias(1)
        @inferred WrenCore._dealias(:hello)
        @inferred WrenCore._dealias("hello")
        @inferred WrenCore._dealias(randn(10))
        @inferred WrenCore._dealias(view(randn(10), 1:5))
        @inferred WrenCore._dealias(view(fill("hi", 10), 1:5))
    end

    @testset "model construction copies args" begin
        @wren function fassume(x)
            a ~ Normal()
            for i in eachindex(x)
                x[i] ~ Normal(a)
            end
        end
        @wren function fobserve(x)
            a ~ Normal()
            for i in eachindex(x)
                @observe x[i] ~ Normal(a)
            end
        end

        x = randn(10)
        xcopy = deepcopy(x)

        @testset "assumed x" begin
            model = fassume(x)
            @test model.args.x !== x # Should be dealiased.
            rand(model) # Run the model
            # Contents should not have been modified, because the model writes
            # into a copy of x.
            @test x == xcopy
        end

        @testset "observed x" begin
            model = fobserve(x)
            @test model.args.x !== x
            rand(model)
            @test x == xcopy
        end

        @testset "conditioned x" begin
            model = fassume(x) | VarNamedTuple(x=x)
            @test model.args.x !== x # Should be dealiased.
            @test model.conditioned[@vn(x)] !== x # Copied on conditioning, like args.
            @test model.conditioned[@vn(x)] == x
            rand(model) # Run the model
            # Contents should not have been modified, because conditioned values are
            # read-only.
            @test x == xcopy
        end
    end

    @testset "model evaluation doesn't usually copy args" begin
        # Unless under Enzyme, but that isn't tested here.
        @wren function fmutates(x)
            x[1] += 1.0
            return x[1]
        end

        X = [0.0] # should never be mutated
        model = fmutates(X)

        @test model.args.x[1] == 0.0
        rand(model)
        @test model.args.x[1] == 1.0
        rand(model)
        @test model.args.x[1] == 2.0
        @test X[1] == 0.0 # Original array should not have been modified
    end

    @testset "AD primal never copies args" begin
        @wren function fmutates(x)
            a ~ Normal() # needs to not be an empty model
            x[1] += 1.0
            return x[1]
        end

        model = fmutates(zeros(1000))
        @testset "$(nameof(typeof(adtype)))" for adtype in (
            AutoForwardDiff(),
            AutoReverseDiff(),
            AutoReverseDiff(; compile=true),
            AutoMooncake(),
            AutoEnzyme(; mode=Forward),
            AutoEnzyme(; mode=Reverse),
        )
            vm = VecModel(model, adtype)

            x_orig = vm.model.args.x[1]
            params = rand(vm)
            @test vm.model.args.x[1] > x_orig

            x_orig = vm.model.args.x[1]
            lp(vm, params)
            @test vm.model.args.x[1] > x_orig

            # For non-Enzyme gradients, it should always be guaranteed that the copy is
            # elided. The problem with testing mutation behaviour is that it's not always
            # visible to the caller: for example Mooncake 'undoes' mutation on the reverse
            # pass. So we have to test this indirectly via allocations.
            #
            # In fact, it's likely that this test will pass for Enzyme as well, but that's
            # only because of `reduce_copies` instead of a guarantee from Wren to not
            # perform the copy. So we don't test that here.
            if !(adtype isa AutoEnzyme)
                bm = @b lp_and_grad($vm, $params)
                @test bm.bytes < sizeof(vm.model.args.x)
            end
        end
    end

    @testset "reduce_copies" begin
        @wren function f_redundant_copy(y)
            a ~ Normal()
            for i in eachindex(y)
                # y[i] ~ Normal(a) would cause allocations O(size(y)), which would
                # make the last test here fail.
                y[i] = a
            end
        end
        Y = randn(1000)
        model = f_redundant_copy(Y)
        @test WrenCore._copied_args(model) == (:y,)

        # Test that without `reduce_copies`, the copy is performed.
        vm = VecModel(model, WrenCore.DEFAULT_AD; reduce_copies=false)
        @test WrenCore._copied_args(vm.model) == (:y,)
        params = rand(vm)
        bm = @b lp_and_grad($vm, $params)
        @test bm.bytes >= sizeof(Y)

        # Test that with `reduce_copies`, the copy is elided.
        vm = VecModel(model, WrenCore.DEFAULT_AD; reduce_copies=true)
        @test WrenCore._copied_args(vm.model) == ()
        bm = @b lp_and_grad($vm, $params)
        @info bm.bytes
        @test bm.bytes < sizeof(Y)
    end

    @testset "copy set is kept when conditioning" begin
        # TODO(penelopeysm): This might change if we make conditioning smarter.
        # https://github.com/JuliaBayes/Wren.jl/issues/139
        @wren function cond_copies(x, y)
            a ~ Normal()
            x[1] ~ Normal(a)
            return x
        end
        model = cond_copies([0.0], [0.0])
        @test WrenCore._copied_args(model) == (:x,)
        conditioned_model = model | VarNamedTuple(a=1.0)
        @test WrenCore._copied_args(conditioned_model) == (:x,)
        @test WrenCore._copied_args(decondition(conditioned_model)) == (:x,)
    end

    @testset "default and keyword arguments are copied like positional ones" begin
        @wren function kw_copies(x, y=[0.0]; z=[0.0], w=[0.0])
            a ~ Normal()
            y[1] ~ Normal(a)
            z[1] ~ Normal(a)
            b ~ Normal(y[1] + z[1])
        end
        model = kw_copies(1.0)
        @test WrenCore._copied_args(model) == (:y, :z)
        # Gradients are only right under Enzyme if the copies are actually made.
        vm_enzyme = VecModel(model, AutoEnzyme(; mode=Reverse); reduce_copies=false)
        vm_fd = VecModel(model, AutoForwardDiff())
        params = rand(vm_enzyme)
        @test lp_and_grad(vm_enzyme, params)[2] ≈ lp_and_grad(vm_fd, params)[2]
    end

    @testset "only arguments that can hold differentiable data are copied" begin
        maybe_dealias(v) =
            WrenCore._maybe_dealias(EvalContext{true}(), Val((:v,)), Val(:v), v)
        # Copied
        v = [1.0, 2.0]
        @test maybe_dealias(v) !== v
        # Not copied
        for v in ([1, 2], [true, false], view([1, 2, 3], 1:2), 1, (1.0, 2.0))
            @test maybe_dealias(v) === v
        end
        # Not in the copy set, so not copied even though it could hold differentiable data
        v = [1.0, 2.0]
        @test WrenCore._maybe_dealias(EvalContext{true}(), Val((:w,)), Val(:v), v) === v

        @testset "integer arrays are still copied on model construction" begin
            @wren function int_arg(z)
                a ~ Normal()
                return z
            end
            z = [1, 2]
            @test int_arg(z).args.z !== z
        end
    end
end

end
