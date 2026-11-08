module WrenDataflowTests

@info "testing dataflow.jl"

using WrenCore
using WrenCore: run_dataflow_analysis
using WrenCore: get_used_symbols, get_lhs_index_symbols
using WrenCore.Expressions: lower_to_wren
using Test

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

@testset "sample lhs reuse analysis" begin
    @testset "basic" begin
        @test first(analyse(@wren internal_to_ir=true function f1()
            x ~ a
            y ~ b
        end)) == Set{Symbol}()

        @test first(analyse(@wren internal_to_ir=true function f2a()
            x ~ a
            y ~ dist(x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f2b()
            x ~ a
            @observe y ~ x
        end)) == Set{Symbol}([:x])

        # Uses prior to sampling don't count
        @test first(analyse(@wren internal_to_ir=true function f3()
            a = y
            y ~ b
        end)) == Set{Symbol}()

        # Any part of the lhs being reused counts as a reuse
        @test first(analyse(@wren internal_to_ir=true function f4()
            x ~ a
            y ~ dist(x[1])
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f5()
            x ~ a
            y = x
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f6()
            x ~ a
            x
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f7()
            x ~ dist(x)
        end)) == Set{Symbol}()

        @test_broken first(analyse(@wren internal_to_ir=true function f8()
            x ~ a
            # In principle, reassigning to x should kill the previous value of x.
            # Wren doesn't detect that yet.
            x = 2
            y ~ dist(x)
        end)) == Set{Symbol}()
    end

    @testset "uses inside expressions" begin
        @test first(analyse(@wren internal_to_ir=true function e1()
            x ~ a
            y ~ dist(x.b)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e2()
            x ~ a
            y ~ dist(f(g(x)))
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e3a()
            x ~ a
            y ~ dist(; b=x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e3b()
            x ~ a
            y ~ dist(; x=b)
        end)) == Set{Symbol}()

        @test first(analyse(@wren internal_to_ir=true function e3c()
            x ~ a
            y ~ dist(x=b)
        end)) == Set{Symbol}()

        @test first(analyse(@wren internal_to_ir=true function e3d()
            x ~ a
            y ~ dist(; x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e4()
            x ~ a
            y ~ dist(x[@vn(b)])
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e5()
            x ~ a
            y ~ dist("$x")
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e6()
            x ~ a
            y ~ dist([x, 1])
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e7()
            x ~ a
            y ~ dist(() -> x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function e8()
            x ~ a
            let z = x
                z
            end
        end)) == Set{Symbol}([:x])
    end

    @testset "closures and inner functions" begin
        # Note: RGF can't define closures over variables outside the model body, but
        # closures over variables defined in the model body are allowed.
        @test first(analyse(@wren internal_to_ir=true function c1()
            x ~ a
            f = () -> x
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function c2()
            x ~ a
            function g()
                return x
            end
        end)) == Set{Symbol}([:x])
    end

    @testset "uses in lhs" begin
        @test first(analyse(@wren internal_to_ir=true function l1()
            x ~ a
            z[x] ~ b
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function l2()
            x ~ a
            @observe z[x] ~ b
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function l3()
            x ~ a
            z[x.b] ~ c
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function l4()
            x[1] ~ a
            x[2] ~ b
        end)) == Set{Symbol}()

        @test first(analyse(@wren internal_to_ir=true function l5()
            x[1] ~ a
            x[x[1]] ~ b
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function l6()
            x[x] ~ a
        end)) == Set{Symbol}()
    end

    @testset "other statements" begin
        @test first(analyse(@wren internal_to_ir=true function o1()
            x ~ a
            @observe y ~ dist(x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function o2()
            x ~ a
            return x
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function o3()
            x ~ a
            return f(x)
        end)) == Set{Symbol}([:x])

        @test first(
            analyse(@wren internal_to_ir=true function o4()
                x ~ a
                @addloglikelihood name = :b f(x)
            end),
        ) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function o5()
            x ~ a
            @inbounds y = x[1]
        end)) == Set{Symbol}([:x])
    end

    @testset "iid" begin
        @test first(analyse(@wren internal_to_ir=true function i1()
            x ~ a
            @iid y ~ dist(x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function i2()
            x ~ a
            @iid size = x y ~ b
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function i3()
            @iid x ~ a
            y ~ dist(x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function i4()
            x ~ a
            @observe @iid y ~ dist(x)
        end)) == Set{Symbol}([:x])
    end

    @testset "batch" begin
        @test first(analyse(@wren internal_to_ir=true function b1()
            x ~ a
            @batch for i in j
                y[i] ~ dist(x[i])
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function b2()
            x ~ a
            @batch for i in x
                y[i] ~ b
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function b3()
            x ~ a
            @batch size = x for i in j
                y[i] ~ b
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function b4()
            @batch for i in j
                y[i] ~ b
            end
            z ~ dist(y)
        end)) == Set{Symbol}([:y])

        @test first(analyse(@wren internal_to_ir=true function b5()
            @batch for i in j
                y[i] ~ b
            end
        end)) == Set{Symbol}()
    end

    @testset "if" begin
        @test first(analyse(@wren internal_to_ir=true function f_if1()
            x ~ a
            if u
                x
            else
                1
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_if2()
            x ~ a
            if x
                foo
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_if3()
            if u
                x ~ a
            end
            y ~ dist(x)
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_if4()
            x ~ a
            if u
                foo
            elseif x
                bar
            end
        end)) == Set{Symbol}([:x])

        @test_broken first(analyse(@wren internal_to_ir=true function f_if5()
            if u
                x ~ a
            else
                y = x
            end
        end)) == Set{Symbol}()
    end

    @testset "for" begin
        @test first(analyse(@wren internal_to_ir=true function f_for1()
            x ~ a
            for i in j
                x
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_for2()
            x ~ a
            for i in 1:10
                y ~ dist(x)
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_for3()
            x ~ a
            for i in x
                y ~ c
            end
        end)) == Set{Symbol}([:x])

        @test_broken first(analyse(@wren internal_to_ir=true function f_for4()
            x ~ a
            # This shouldn't count because the loop index `x` will shadow the top-level `x`.
            # Wren currently conservatively reports that `x` is reused, but in principle it
            # shouldn't be.
            for x in y
                z ~ c
            end
        end)) == Set{Symbol}()

        @test first(analyse(@wren internal_to_ir=true function f_for_backedge(a)
            # See discussion in https://github.com/JuliaBayes/Wren.jl/pull/124
            for i in j
                z = a
                a ~ b
            end
        end)) == Set{Symbol}([:a])

        @test first(analyse(@wren internal_to_ir=true function f_for_nested_backedge()
            for i in j
                for k in l
                    z = a
                end
                a ~ b
            end
        end)) == Set{Symbol}([:a])

        @test first(analyse(@wren internal_to_ir=true function f_for_self_backedge()
            for i in j
                x ~ dist(x)
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_for_no_backedge()
            for i in j
                a ~ b
            end
        end)) == Set{Symbol}()
    end

    @testset "while" begin
        @test first(analyse(@wren internal_to_ir=true function f_while1()
            x ~ a
            while u
                y ~ dist(x)
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_while_backedge()
            while u
                z = a
                a ~ b
            end
        end)) == Set{Symbol}([:a])

        @test first(analyse(@wren internal_to_ir=true function f_while_cond()
            while a
                a ~ b
            end
        end)) == Set{Symbol}([:a])

        @test first(analyse(@wren internal_to_ir=true function f_while_self_backedge()
            while u
                x ~ dist(x)
            end
        end)) == Set{Symbol}([:x])

        @test first(analyse(@wren internal_to_ir=true function f_while_no_backedge()
            while u
                a ~ b
            end
        end)) == Set{Symbol}()
    end
end

@testset "arg copy analysis" begin
    @testset "basic" begin
        @test last(analyse(@wren internal_to_ir=true function f1(a)
            x ~ a
        end)) == Set{Symbol}([:a])

        @test_broken last(analyse(@wren internal_to_ir=true function f2(a)
            # Reassigning to the entire variable `a` should kill the previous value, but
            # Wren doesn't detect that.
            a = x
        end)) == Set{Symbol}()

        @test last(analyse(@wren internal_to_ir=true function f3(a)
            # Very conservative, can be improved - see #150
            x = a
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function f4(a)
            # Very conservative, can be improved - see #150
            x = f(a)
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function f5(a)
            # Very conservative, can be improved - see #150
            x ~ Normal(a)
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function f6(a)
            a ~ Normal()
        end)) == Set{Symbol}()

        @test last(analyse(@wren internal_to_ir=true function f7(a)
            @observe a ~ Normal()
        end)) == Set{Symbol}()
    end

    @testset "genuine cases where it must be copied" begin
        @test last(analyse(@wren internal_to_ir=true function g1(a)
            d ~ y       # `d` is differentiable
            c ~ f(a, d) # potentially copied into `a` and read via `c`
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function g2(a)
            a ~ y     # `a` is differentiable
            b ~ f(a)  # directly read out
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function g3(a)
            a ~ y                   # `a` is differentiable
            @addloglikelihood f(a)  # directly read out
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function g4(a)
            # This was the original trigger for adding copies, so if this fails the whole
            # thing is broken! https://github.com/JuliaBayes/Wren.jl/issues/69
            @batch for i in 1:N
                a[i] ~ d
                b[i] ~ dist(a[i])
            end
        end)) == Set{Symbol}([:a])
    end

    @testset "multiple samples / observes" begin
        @test last(analyse(@wren internal_to_ir=true function ffor1(a)
            for i in 1:N
                a[i] ~ dist
            end
        end)) == Set{Symbol}()

        @test last(analyse(@wren internal_to_ir=true function ffor2(a)
            # Very conservative, can be improved - see #150
            for i in 1:eachindex(a)
                a[i] ~ dist
            end
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function ffor3(a)
            for i in 1:N
                @observe a[i] ~ dist
            end
        end)) == Set{Symbol}()

        @test last(analyse(@wren internal_to_ir=true function ffor4(a)
            # Very conservative, can be improved - see #150
            for i in 1:eachindex(a)
                @observe a[i] ~ dist
            end
        end)) == Set{Symbol}([:a])
    end

    @testset "iid" begin
        @test last(analyse(@wren internal_to_ir=true function i1(a)
            # Very conservative, can be improved - see #150
            @iid x ~ dist(a)
        end)) == Set{Symbol}([:a])

        @test last(analyse(@wren internal_to_ir=true function i2(a)
            # This is very conservative as well, but not related to #150
            @iid x ~ dist size=a
        end)) == Set{Symbol}([:a])
    end

    @testset "batch" begin
        @test last(analyse(@wren internal_to_ir=true function b2(a)
            @batch for i in 1:N
                a[i] ~ d
            end
        end)) == Set{Symbol}()

        @test last(analyse(@wren internal_to_ir=true function b3(a)
            # Very conservative, can be improved - see #150
            @batch for i in 1:N
                b[i] ~ a[i]
            end
        end)) == Set{Symbol}([:a])
    end
end


_samples(stmts) = filter(s -> s isa WrenCore.Sample, stmts)

@testset "rewriting stage of dataflow analysis" begin
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

    @testset "generated quantity" begin
        @test first(analyse(@wren internal_to_ir=true function r8()
            x ~ Normal()
            @gq y = x
        end)) == Set{Symbol}([:x])

        ir, args = @wren internal_to_ir=true function r9()
            x ~ Normal()
            @gq x
        end
        ir, _ = run_dataflow_analysis(ir, args)
        @test only(filter(s -> s isa WrenCore.Sample, ir)).lhs_is_reused

        ir, args = @wren internal_to_ir=true function r10(x, y)
            a ~ Normal()
            @gq mu = a .+ x
            @observe y ~ MvNormal(mu, I)
            @gq rss = sum(abs2, y .- mu)
        end
        ir, _ = run_dataflow_analysis(ir, args)
        gqs = filter(s -> s isa WrenCore.GeneratedQuantity, ir)
        @test [(s.name, s.name_is_reused) for s in gqs] == [(:mu, true), (:rss, false)]

        @test first(analyse(@wren internal_to_ir=true function r11()
            a ~ Normal()
            @gq b = a
            @gq c = b
        end)) == Set{Symbol}([:a])

        ir, args = @wren internal_to_ir=true function r12()
            a ~ Normal()
            @gq b = a
            @gq c = b
            @gq d = c
            @gq e = a
            f ~ Normal(e)
        end
        ir, _ = run_dataflow_analysis(ir, args)
        gqs = filter(s -> s isa WrenCore.GeneratedQuantity, ir)
        @test [(s.name, s.name_is_reused, s.dependent_gqs) for s in gqs] == [
            (:b, false, [:c, :d]),
            (:c, false, [:d]),
            (:d, false, Symbol[]),
            (:e, true, Symbol[]),
        ]
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
