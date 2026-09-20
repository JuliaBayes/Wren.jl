module WrenExprTests

@info "testing expr.jl"

using WrenCore
using Test

# Normalise ASTs (mostly by removing line number nodes, which are not relevant for semantics).
_normalise_ast!(ex) = ex
function _normalise_ast!(ex::Expr)
    if ex.head === :block || ex.head === :quote || ex.head === :toplevel
        filter!(ex.args) do x
            !(isa(x, Expr) && x.head === :line || isa(x, LineNumberNode))
        end
    end
    # macrocall's apparently _need_ to have a line number node or something in that
    # position, so we can't remove it altogether -- but we can replace all such nodes with
    # `nothing` so that they compare equal. See: https://stackoverflow.com/a/53285644
    if ex.head === :macrocall
        new_args = map(ex.args) do x
            x isa LineNumberNode ? nothing : x
        end
        ex.args = new_args
    end
    for subex in ex.args
        subex isa Expr && _normalise_ast!(subex)
    end
    return ex
end

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
            f(expr) = WrenCore.replace_loop_idx_with_iters(expr, :i, :(1:J), false)
            BE = WrenCore.BatchedExpr
            @test f(:x) == BE(:x, false, nothing, nothing)
            @test f(:i) == BE(:(1:J), true, nothing, nothing)
            @test f(:(x[i])) == BE(:(x[1:J]), true, 1, 1)
            @test f(:(x[i, j])) == BE(:(x[1:J, j]), true, 1, 2)
            @test f(:(x[j, i])) == BE(:(x[j, 1:J]), true, 2, 2)
            @test f(:(x[1, 2, 3, i, 5, 6])) == BE(:(x[1, 2, 3, 1:J, 5, 6]), true, 4, 6)
            @test f(:(x[j, k])) == BE(:(x[j, k]), false, nothing, nothing)

            @test f(:(x.a)) == BE(:(x.a), false, nothing, nothing)
            @test f(:(x.i)) == BE(:(x.i), false, nothing, nothing) # NOTE: don't expand a field called `i`!
            @test f(:(x.a[i])) == BE(:(x.a[1:J]), true, 1, 1)
            @test f(:(x.a[i, j])) == BE(:(x.a[1:J, j]), true, 1, 2)
            @test f(:(x.a[j, i])) == BE(:(x.a[j, 1:J]), true, 2, 2)

            @test f(:(func(x))) == BE(:(func(x)), false, nothing, nothing)
            @test f(:(func(x.i))) == BE(:(func(x.i)), false, nothing, nothing)

            # These should all error as we can't safely batch them :(.
            @test_throws WrenCore.CannotBatch f(:(func(x[i])))
            @test_throws WrenCore.CannotBatch f(:(func(x[i, j])))
            @test_throws WrenCore.CannotBatch f(:(i + 1))
            @test_throws WrenCore.CannotBatch f(:(x[i+1]))
            @test_throws WrenCore.CannotBatch f(:(x[-i]))
            @test_throws WrenCore.CannotBatch f(:(1 + x[i]))
            @test_throws WrenCore.CannotBatch f(:((1 + x[i]) * cnst))
            @test_throws WrenCore.CannotBatch f(:((1 + x[i]) * y[i]))
            @test_throws WrenCore.CannotBatch f(:(func(1 + x[i])))
            @test_throws WrenCore.CannotBatch f(:(x[i].a))
        end

        @testset "batch_expr" begin
            function test_batch(expr, expected, expected_info_log)
                if expected_info_log === nothing
                    @test_logs WrenCore.batch_expr(expr, :i, :(1:J), :s)
                else
                    @test_logs (:info, expected_info_log) WrenCore.batch_expr(
                        expr,
                        :i,
                        :(1:J),
                        :s,
                    )
                end
                actual = WrenCore.batch_expr(expr, :i, :(1:J), :s)

                if _normalise_ast!(actual) != _normalise_ast!(expected)
                    dump(actual)
                    dump(expected)
                end
                @test _normalise_ast!(actual) == _normalise_ast!(expected)
            end

            test_batch(
                :(Normal(x[i])),
                :($(WrenCore.batched)(
                    1:J,
                    $(Val){Normal}(),
                    $(WrenCore.Indexed){1,1}(@view(x[1:J])),
                )),
                nothing,
            )

            test_batch(
                :(Normal(x[i], y[i])),
                :($(WrenCore.batched)(
                    1:J,
                    $(Val){Normal}(),
                    $(WrenCore.Indexed){1,1}(@view(x[1:J])),
                    $(WrenCore.Indexed){1,1}(@view(y[1:J])),
                )),
                nothing,
            )

            test_batch(
                :(Normal(x[i], sigma)),
                :($(WrenCore.batched)(
                    1:J,
                    $(Val){Normal}(),
                    $(WrenCore.Indexed){1,1}(@view(x[1:J])),
                    sigma,
                )),
                nothing,
            )

            test_batch(
                :(Normal(x[i, 1])),
                :($(WrenCore.batched)(
                    1:J,
                    $(Val){Normal}(),
                    $(WrenCore.Indexed){1,2}(@view(x[1:J, 1])),
                )),
                nothing,
            )

            test_batch(
                :(Normal(x[2, i])),
                :($(WrenCore.batched)(
                    1:J,
                    $(Val){Normal}(),
                    $(WrenCore.Indexed){2,2}(@view(x[2, 1:J])),
                )),
                nothing,
            )

            test_batch(
                :(Normal(x[i, i])),
                :($(WrenCore.batched)(
                    1:J,
                    $(Val){Normal}(),
                    $(WrenCore.Indexed){$nothing,$nothing}(
                        $(getindex).($(Ref)(x), 1:J, 1:J),
                    ),
                )),
                nothing,
            )

            # No-batching cases
            test_batch(
                :(Normal(exp(x[i]))),
                :($(WrenCore.independent)([Normal(exp(x[i])) for i in 1:J])),
                r"is not a simple indexing expression",
            )
            test_batch(
                :(Normal(x[i+1])),
                :($(WrenCore.independent)([Normal(x[i+1]) for i in 1:J])),
                r"that is not the loop index",
            )
            test_batch(
                :(Normal(; x=i)),
                :($(WrenCore.independent)([Normal(; x=i) for i in 1:J])),
                r"has keyword arguments",
            )
            test_batch(
                :(
                    begin
                        i
                    end
                ),
                :($(WrenCore.independent)([
                    begin
                        i
                    end for i in 1:J
                ])),
                r"not a function call",
            )
        end

    end
end

end # module WrenExprTests
