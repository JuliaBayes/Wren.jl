module WrenExprTests

@info "testing expr.jl"

using WrenCore
using Test

# todo: This needs to be greatly expanded for our special nodes...

@testset "expr unit tests" begin
    @testset "iid" begin
        lower(expr) = only(
            filter(
                e -> e isa WrenCore.AbstractWrenExpr,
                WrenCore.to_wren_exprs(expr, @__MODULE__),
            ),
        )
        # `@iid` is a marker: it wraps the tilde-statement without rewriting it.
        s = lower(:(@iid x ~ Normal()))
        @test s isa WrenCore.IID{<:WrenCore.Sample}
        @test s.tilde.lhs == :x

        # Either order of `@iid` and `@observe` gives the same node.
        for expr in (:(@observe @iid y ~ Normal(mu)), :(@iid @observe y ~ Normal(mu)))
            o = lower(expr)
            @test o isa WrenCore.IID{<:WrenCore.Observe}
            @test o.tilde.lhs == :y
        end

        @test_throws ErrorException lower(:(@iid x))
        @test_throws ErrorException lower(:(@iid x ~ Normal() Normal()))
        @test_throws ErrorException lower(:(@observe @iid x))
        @test_throws ErrorException lower(:(@iid @observe x))
        # Not in statement position
        @test_throws ErrorException lower(:(z = @iid x ~ Normal()))
    end

    @testset "occurs_in_expr" begin
        f(expr) = WrenCore.occurs_in_expr(expr, :i)
        @test f(:x) == false
        @test f(:i) == true
        @test f(:(x[i])) == true
        @test f(:(x[i, j])) == true
        @test f(:(x[j, i])) == true
        @test f(:(x[j, k])) == false
        @test f(:(x.a)) == false
        @test f(:(x.i)) == false
        @test f(:(x.a[i])) == true
        @test f(:(i + 1)) == true
    end

    @testset "batch" begin
        @testset "occurs_in_kwargs" begin
            function f(expr)
                if expr.head != :call ||
                   length(expr.args) < 3 ||
                   !(expr.args[2] isa Expr) ||
                   expr.args[2].head != :parameters
                    error("expr must be a call with kwargs")
                end
                return WrenCore.occurs_in_kwargs(expr.args[2], :i)
            end
            @test !f(:(func(x; a=1, b=2)))
            @test !f(:(func(x; i=1)))
            @test f(:(func(x; x=i)))
            @test f(:(func(x; i)))
            @test f(:(func(x; a, i)))
            @test f(:(func(x; x=f(i))))
            @test !f(:(func(x; x=a.i)))
        end

        @testset "replace_loop_idx_with_iters" begin
            f(expr) = WrenCore.replace_loop_idx_with_iters(expr, :i, :(1:J))
            @test f(:x) == (:x, false)
            @test f(:i) == (:(1:J), true)
            @test f(:(x[i])) == (:(x[1:J]), true)
            @test f(:(x[i, j])) == (:(x[1:J, j]), true)
            @test f(:(x[j, i])) == (:(x[j, 1:J]), true)
            @test f(:(x[j, k])) == (:(x[j, k]), false)
            @test f(:(x.a)) == (:(x.a), false)
            @test f(:(x.i)) == (:(x.i), false) # NOTE: don't expand a field called `i`!
            @test f(:(x.a[i])) == (:(x.a[1:J]), true)
            @test f(:(x.a[i, j])) == (:(x.a[1:J, j]), true)
            @test f(:(func(x))) == (:(func(x)), false)
            @test f(:(func(x.i))) == (:(func(x.i)), false)

            # Broadcasting operations where needed...
            @test f(:(func(x[i]))) == (:(func.(x[1:J])), true)
            @test f(:(func(x[i, j]))) == (:(func.(x[1:J, j])), true)
            @test f(:(func(x.a[i]))) == (:(func.(x.a[1:J])), true)
            @test f(:(i + 1)) == (:((1:J) .+ 1), true)
            @test f(:(x[i+1])) == (:(x[(1:J) .+ 1]), true)
            @test f(:(x[-i])) == (:(x[.-(1:J)]), true)
            @test f(:(1 + x[i])) == (:(1 .+ x[1:J]), true)
            @test f(:((1 + x[i]) * cnst)) == (:((1 .+ x[1:J]) .* cnst), true)
            @test f(:((1 + x[i]) * y[i])) == (:((1 .+ x[1:J]) .* y[1:J]), true)
            @test f(:(func(1 + x[i]))) == (:(func.(1 .+ x[1:J])), true)

            @test_broken false
            # The following expression should error:
            # f(:(x[i].a))
        end
    end
end

end # module WrenExprTests
