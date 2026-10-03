module WrenDataflowTests

@info "testing dataflow.jl"

using WrenCore
using WrenCore: run_dataflow_analysis
using WrenCore: get_used_symbols, get_lhs_index_symbols
using WrenCore.Expressions: lower_to_wren
using Test

# Run the dataflow analysis on the output of `@wren internal_to_ir=true`. Returns a tuple of
# `(required_syms, args_to_copy)`.
analyse((ir, args)) = WrenCore.get_required_syms_and_args(ir, args)

@testset "get_used_symbols" begin
    # Check both the raw Julia expression and its lowered form.
    function test_used(expr, expected)
        @test get_used_symbols(expr) == Set{Symbol}(expected)
        @test get_used_symbols(lower_to_wren(expr, @__MODULE__)) == Set{Symbol}(expected)
    end

    @testset "leaves" begin
        test_used(:x, [:x])
        test_used(1, [])
        test_used("s", [])
        test_used(QuoteNode(:x), [])
        test_used(nothing, [])
    end

    @testset "calls" begin
        test_used(:(f()), [:f])
        test_used(:(f(x, y)), [:f, :x, :y])
        test_used(:(f(g(x))), [:f, :g, :x])
        test_used(:(x + y), [:+, :x, :y])
        test_used(:(f.(x)), [:f, :x])
    end

    @testset "keyword arguments" begin
        test_used(:(f(; a=x)), [:f, :x])
        test_used(:(f(a=x)), [:f, :x])
        test_used(:(f(; x)), [:f, :x])
        test_used(:(f(x; x=y)), [:f, :x, :y])
    end

    @testset "indexing and properties" begin
        test_used(:(x[i]), [:x, :i])
        test_used(:(x.a), [:x])
        test_used(:(x.a[i].b), [:x, :i])
        test_used(:(x[@vn(a)]), [:x, :a])
    end

    @testset "collections" begin
        test_used(:([]), [])
        test_used(:(()), [])
        test_used(:([x, 1]), [:x])
        test_used(:((x, y)), [:x, :y])
        test_used(:("$x"), [:x])
    end

    @testset "control flow and closures" begin
        test_used(:(x ? y : z), [:x, :y, :z])
        test_used(:(() -> x), [:x])
        test_used(:(map(i -> x * i, y)), [:map, :i, :x, :*, :y])
    end
end

@testset "get_lhs_index_symbols" begin
    @test get_lhs_index_symbols(:x) == Set{Symbol}()
    @test get_lhs_index_symbols(:(x[i])) == Set{Symbol}([:i])
    @test get_lhs_index_symbols(:(x[i, j])) == Set{Symbol}([:i, :j])
    @test get_lhs_index_symbols(:(x.a)) == Set{Symbol}()
    @test get_lhs_index_symbols(:(x[i].a[j])) == Set{Symbol}([:i, :j])
    @test get_lhs_index_symbols(:(x[f(i)])) == Set{Symbol}([:f, :i])
    @test get_lhs_index_symbols(:(x[x[1]])) == Set{Symbol}([:x])
    @test get_lhs_index_symbols(:(x[1[x]])) == Set{Symbol}([:x])
end

@testset "dataflow analysis" begin
    @testset "basic" begin
        @test analyse(@wren internal_to_ir=true function f1(a, b)
            x ~ a
            y ~ b
        end) == (Set{Symbol}(), Set{Symbol}([:a, :b]))

        @test analyse(@wren internal_to_ir=true function f2a(a)
            x ~ a
            y ~ dist(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function f2b(a, y)
            x ~ a
            @observe y ~ x
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        # Uses prior to sampling don't count
        @test analyse(@wren internal_to_ir=true function f3(y, b)
            a = y
            y ~ b
        end) == (Set{Symbol}(), Set{Symbol}([:y, :b]))

        # Any part of the lhs being reused counts as a reuse
        @test analyse(@wren internal_to_ir=true function f4(a)
            x ~ a
            y ~ dist(x[1])
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function f5(a)
            x ~ a
            y = x
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function f6(a)
            x ~ a
            x
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function f7(x)
            x ~ dist(x)
        end) == (Set{Symbol}(), Set{Symbol}([:x]))

        required, to_copy = analyse(@wren internal_to_ir=true function f8(a)
            x ~ a
            # In principle, reassigning to x should kill the previous value of x.
            # Wren doesn't detect that yet.
            x = 2
            y ~ dist(x)
        end)

        @test_broken required == Set{Symbol}()

        @test to_copy == Set{Symbol}([:a])
    end

    @testset "uses inside expressions" begin
        @test analyse(@wren internal_to_ir=true function e1(a)
            x ~ a
            y ~ dist(x.b)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e2(a)
            x ~ a
            y ~ dist(f(g(x)))
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e3a(a)
            x ~ a
            y ~ dist(; b=x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e3b(a, b)
            x ~ a
            y ~ dist(; x=b)
        end) == (Set{Symbol}(), Set{Symbol}([:a, :b]))

        @test analyse(@wren internal_to_ir=true function e3c(a, b)
            x ~ a
            y ~ dist(x=b)
        end) == (Set{Symbol}(), Set{Symbol}([:a, :b]))

        @test analyse(@wren internal_to_ir=true function e3d(a)
            x ~ a
            y ~ dist(; x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e4(a)
            x ~ a
            y ~ dist(x[@vn(b)])
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e5(a)
            x ~ a
            y ~ dist("$x")
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e6(a)
            x ~ a
            y ~ dist([x, 1])
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e7(a)
            x ~ a
            y ~ dist(() -> x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function e8(a)
            x ~ a
            let z = x
                z
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))
    end

    @testset "closures and inner functions" begin
        # Note: RGF can't define closures over variables outside the model body, but
        # closures over variables defined in the model body are allowed.
        @test analyse(@wren internal_to_ir=true function c1(a)
            x ~ a
            f = () -> x
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function c2(a)
            x ~ a
            function g()
                return x
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))
    end

    @testset "uses in lhs" begin
        @test analyse(@wren internal_to_ir=true function l1(a, b, z)
            x ~ a
            z[x] ~ b
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :b]))

        @test analyse(@wren internal_to_ir=true function l2(a, b, z)
            x ~ a
            @observe z[x] ~ b
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :b]))

        @test analyse(@wren internal_to_ir=true function l3(a, c, z)
            x ~ a
            z[x.b] ~ c
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :c]))

        @test analyse(@wren internal_to_ir=true function l4(x, a, b)
            x[1] ~ a
            x[2] ~ b
        end) == (Set{Symbol}(), Set{Symbol}([:a, :b]))

        @test analyse(@wren internal_to_ir=true function l5(x, a, b)
            x[1] ~ a
            x[x[1]] ~ b
        end) == (Set{Symbol}([:x]), Set{Symbol}([:x, :a, :b]))

        @test analyse(@wren internal_to_ir=true function l6(x, a)
            x[x] ~ a
        end) == (Set{Symbol}(), Set{Symbol}([:x, :a]))
    end

    @testset "other statements" begin
        @test analyse(@wren internal_to_ir=true function o1(a, y)
            x ~ a
            @observe y ~ dist(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function o2(a)
            x ~ a
            return x
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function o3(a)
            x ~ a
            return f(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function o4(a)
            x ~ a
            @addloglikelihood name = :b f(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function o5(a)
            x ~ a
            @inbounds y = x[1]
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))
    end

    @testset "iid" begin
        @test analyse(@wren internal_to_ir=true function i1(a)
            x ~ a
            @iid y ~ dist(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function i2(a, b)
            x ~ a
            @iid size = x y ~ b
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :b]))

        @test analyse(@wren internal_to_ir=true function i3(a)
            @iid x ~ a
            y ~ dist(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function i4(a, y)
            x ~ a
            @observe @iid y ~ dist(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))
    end

    @testset "batch" begin
        @test analyse(@wren internal_to_ir=true function b1(a, j, y)
            x ~ a
            @batch for i in j
                y[i] ~ dist(x[i])
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :j]))

        @test analyse(@wren internal_to_ir=true function b2(a, y)
            x ~ a
            @batch for i in x
                y[i] ~ b
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function b3(a, b, j, y)
            x ~ a
            @batch size = x for i in j
                y[i] ~ b
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :b, :j]))

        @test analyse(@wren internal_to_ir=true function b4(b, j, y)
            @batch for i in j
                y[i] ~ b
            end
            z ~ dist(y)
        end) == (Set{Symbol}([:y]), Set{Symbol}([:b, :j, :y]))

        @test analyse(@wren internal_to_ir=true function b5(b, j, y)
            @batch for i in j
                y[i] ~ b
            end
        end) == (Set{Symbol}(), Set{Symbol}([:b, :j]))
    end

    @testset "if" begin
        @test analyse(@wren internal_to_ir=true function f_if1(a, u)
            x ~ a
            if u
                x
            else
                1
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :u]))

        @test analyse(@wren internal_to_ir=true function f_if2(a, foo)
            x ~ a
            if x
                foo
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :foo]))

        @test analyse(@wren internal_to_ir=true function f_if3(a, u)
            if u
                x ~ a
            end
            y ~ dist(x)
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :u]))

        @test analyse(@wren internal_to_ir=true function f_if4(a, u, foo, bar)
            x ~ a
            if u
                foo
            elseif x
                bar
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :u, :foo, :bar]))

        required, to_copy = analyse(@wren internal_to_ir=true function f_if5(a, u)
            if u
                x ~ a
            else
                y = x
            end
        end)

        @test_broken required == Set{Symbol}()

        @test to_copy == Set{Symbol}([:a, :u])
    end

    @testset "for" begin
        @test analyse(@wren internal_to_ir=true function f_for1(a, j)
            x ~ a
            for i in j
                x
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :j]))

        @test analyse(@wren internal_to_ir=true function f_for2(a)
            x ~ a
            for i in 1:10
                y ~ dist(x)
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a]))

        @test analyse(@wren internal_to_ir=true function f_for3(a, c)
            x ~ a
            for i in x
                y ~ c
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :c]))

        required, to_copy = analyse(@wren internal_to_ir=true function f_for4(a, y, c)
            x ~ a
            # This shouldn't count because the loop index `x` will shadow the top-level `x`.
            # Wren currently conservatively reports that `x` is reused, but in principle it
            # shouldn't be.
            for x in y
                z ~ c
            end
        end)

        @test_broken required == Set{Symbol}()

        @test to_copy == Set{Symbol}([:a, :y, :c])

        @test analyse(@wren internal_to_ir=true function f_for_backedge(a, b, j)
            # See discussion in https://github.com/JuliaBayes/Wren.jl/pull/124
            for i in j
                z = a
                a ~ b
            end
        end) == (Set{Symbol}([:a]), Set{Symbol}([:a, :b, :j]))

        @test analyse(@wren internal_to_ir=true function f_for_nested_backedge(b, j, l)
            for i in j
                for k in l
                    z = a
                end
                a ~ b
            end
        end) == (Set{Symbol}([:a]), Set{Symbol}([:b, :j, :l]))

        @test analyse(@wren internal_to_ir=true function f_for_self_backedge(j)
            for i in j
                x ~ dist(x)
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:j]))

        @test analyse(@wren internal_to_ir=true function f_for_no_backedge(b, j)
            for i in j
                a ~ b
            end
        end) == (Set{Symbol}(), Set{Symbol}([:b, :j]))
    end

    @testset "while" begin
        @test analyse(@wren internal_to_ir=true function f_while1(a, u)
            x ~ a
            while u
                y ~ dist(x)
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:a, :u]))

        @test analyse(@wren internal_to_ir=true function f_while_backedge(b, u)
            while u
                z = a
                a ~ b
            end
        end) == (Set{Symbol}([:a]), Set{Symbol}([:b, :u]))

        @test analyse(@wren internal_to_ir=true function f_while_cond(b)
            while a
                a ~ b
            end
        end) == (Set{Symbol}([:a]), Set{Symbol}([:b]))

        @test analyse(@wren internal_to_ir=true function f_while_self_backedge(u)
            while u
                x ~ dist(x)
            end
        end) == (Set{Symbol}([:x]), Set{Symbol}([:u]))

        @test analyse(@wren internal_to_ir=true function f_while_no_backedge(b, u)
            while u
                a ~ b
            end
        end) == (Set{Symbol}(), Set{Symbol}([:b, :u]))
    end

    @testset "argument copying" begin
        @test analyse(@wren internal_to_ir=true function ac_unused(y)
            x ~ a
        end) == (Set{Symbol}(), Set{Symbol}())

        # Written to, then read from: needs a copy.
        @test analyse(@wren internal_to_ir=true function ac_write_read(y)
            y[1] ~ a
            @observe y[1] ~ b
        end) == (Set{Symbol}([:y]), Set{Symbol}([:y]))

        # Read from, then written to: no copy. This means that the write will be visible on
        # the next model evaluation, which is intended (see the docs on mutability).
        @test analyse(@wren internal_to_ir=true function ac_read_write(y)
            @observe y[1] ~ a
            y[1] ~ b
        end) == (Set{Symbol}(), Set{Symbol}())

        # Inside loops, a read can come after a write from the previous iteration.
        @test analyse(@wren internal_to_ir=true function ac_read_write_for(y, j)
            for i in j
                @observe y[i] ~ a
                y[i] ~ b
            end
        end) == (Set{Symbol}([:y]), Set{Symbol}([:y, :j]))

        @test analyse(@wren internal_to_ir=true function ac_read_write_while(y, u)
            while u
                @observe y[1] ~ a
                y[1] ~ b
            end
        end) == (Set{Symbol}([:y]), Set{Symbol}([:y, :u]))

        # The whole LHS of an observe statement is only read from, including indices. (Side
        # effects in the LHS are forbidden.) This is unlike Sample, where indices are
        # treated as general uses.
        @test analyse(@wren internal_to_ir=true function ac_observe_index(y, i)
            @observe y[i] ~ a
        end) == (Set{Symbol}(), Set{Symbol}())

        @test analyse(@wren internal_to_ir=true function ac_sample_index(y, i)
            y[i] ~ a
        end) == (Set{Symbol}(), Set{Symbol}([:i]))

        # Sampling a top-level argument is only a write...
        @test analyse(@wren internal_to_ir=true function ac_toplevel(x)
            x ~ a
        end) == (Set{Symbol}(), Set{Symbol}())

        # ...but returning it counts as a read, since the return value may be used by a
        # parent model (if this is a submodel) or by the user.
        @test analyse(@wren internal_to_ir=true function ac_toplevel_return(x)
            x ~ a
            return x
        end) == (Set{Symbol}([:x]), Set{Symbol}([:x]))

        @test analyse(@wren internal_to_ir=true function ac_batch_observe(y, j)
            @batch for i in j
                @observe y[i] ~ a
            end
        end) == (Set{Symbol}(), Set{Symbol}([:j]))
    end
end

_samples(stmts) = filter(s -> s isa WrenCore.Sample, stmts)

@testset "rewriting samples with lhs_is_reused" begin
    @testset "top level" begin
        ir, args = @wren internal_to_ir=true function r1()
            x ~ a
            y ~ dist(x)
        end
        ir, _ = run_dataflow_analysis(ir, args)
        x_stmt, y_stmt = _samples(ir)
        @test x_stmt.lhs_is_reused
        @test !y_stmt.lhs_is_reused
    end

    @testset "inside for and if" begin
        ir, args = @wren internal_to_ir=true function r2()
            x ~ a
            for i in j
                y ~ dist(x)
                if u
                    z ~ dist(y)
                end
            end
        end
        ir, _ = run_dataflow_analysis(ir, args)
        x_stmt = only(_samples(ir))
        for_body = only(filter(s->s isa WrenCore.WrenExpr, ir)).args[2]
        y_stmt = only(_samples(for_body.args))
        if_body = only(filter(s->s isa WrenCore.WrenExpr, for_body.args)).args[2]
        z_stmt = only(_samples(if_body.args))
        @test x_stmt.lhs_is_reused
        @test y_stmt.lhs_is_reused
        @test !z_stmt.lhs_is_reused
    end

    @testset "iid" begin
        ir, args = @wren internal_to_ir=true function r3()
            @iid x ~ a
            @iid y ~ dist(x)
        end
        ir, _ = run_dataflow_analysis(ir, args)
        x_stmt, y_stmt = filter(s -> s isa WrenCore.IID, ir)
        @test x_stmt.tilde.lhs_is_reused
        @test !y_stmt.tilde.lhs_is_reused
    end

    @testset "batch" begin
        ir, args = @wren internal_to_ir=true function r4()
            @batch for i in j
                y[i] ~ b
            end
            @batch for i in j
                z[i] ~ b
            end
            w ~ dist(z)
        end
        ir, _ = run_dataflow_analysis(ir, args)
        y_batch, z_batch = filter(s -> s isa WrenCore.Batch, ir)
        y_stmt = only(_samples(y_batch.body.args))
        z_stmt = only(_samples(z_batch.body.args))
        @test !y_stmt.lhs_is_reused
        @test z_stmt.lhs_is_reused
    end

    @testset "return" begin
        ir, args = @wren internal_to_ir=true function r6()
            return x ~ a
        end
        ir, _ = run_dataflow_analysis(ir, args)
        @test !only(filter(s -> s isa WrenCore.ReturnStatement, ir)).expr.lhs_is_reused
    end

    @testset "addloglikelihood" begin
        ir, args = @wren internal_to_ir=true function r7()
            @addloglikelihood (x ~ a)
        end
        ir, _ = run_dataflow_analysis(ir, args)
        @test !only(filter(s -> s isa WrenCore.AddLogLikelihood, ir)).expr.lhs_is_reused
    end

    @testset "lhs_top_level_symbol" begin
        ir, args = @wren internal_to_ir=true function r5()
            x ~ a
            y[1] ~ b
            z.c ~ d
            w[1].e[2] ~ f
        end
        ir, _ = run_dataflow_analysis(ir, args)
        @test map(s -> s.lhs_top_level_symbol, _samples(ir)) == [:x, :y, :z, :w]
    end
end

end # module WrenDataflowTests
