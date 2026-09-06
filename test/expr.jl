module WrenExprTests

@info "testing expr.jl"

using Wren
using Test

# todo: This needs to be greatly expanded for our special nodes...

@testset "expr unit tests" begin
    @testset "plate" begin
        @testset "replace_loop_idx_with_iters" begin
            f(expr) = Wren.replace_loop_idx_with_iters(expr, :i, :(1:J))
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
