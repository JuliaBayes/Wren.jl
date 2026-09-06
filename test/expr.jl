module WrenExprTests

@info "testing expr.jl"

using Wren
using Test

# Normalise ASTs (mostly by removing line number nodes, which are not relevant for
# semantics). This function is taken from JuliaFormatter
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
            filter(e -> e isa Wren.AbstractWrenExpr, Wren.to_wren_exprs(expr, @__MODULE__)),
        )
        roundtrip(expr) = _normalise_ast!(Wren._wren_expr_to_model_expr(lower(expr)))

        # `@iid` is a marker: it wraps the tilde-statement without rewriting it.
        s = lower(:(@iid x ~ Normal()))
        @test s isa Wren.IID{<:Wren.Sample}
        @test s.tilde.lhs == :x
        @test Wren._wren_expr_to_model_expr(s.tilde.rhs) == :(Normal())
        @test roundtrip(:(@iid x ~ Normal())) == _normalise_ast!(:(@iid x ~ Normal()))

        # Either order of `@iid` and `@observe` gives the same node.
        for expr in (:(@observe @iid y ~ Normal(mu)), :(@iid @observe y ~ Normal(mu)))
            o = lower(expr)
            @test o isa Wren.IID{<:Wren.Observe}
            @test o.tilde.lhs == :y
            @test Wren._wren_expr_to_model_expr(o.tilde.rhs) == :(Normal(mu))
            @test roundtrip(expr) == _normalise_ast!(:(@iid @observe y ~ Normal(mu)))
        end

        @test_throws ErrorException lower(:(@iid x))
        @test_throws ErrorException lower(:(@iid x ~ Normal() Normal()))
        @test_throws ErrorException lower(:(@observe @iid x))
        @test_throws ErrorException lower(:(@iid @observe x))
        # Not in statement position
        @test_throws ErrorException lower(:(z = @iid x ~ Normal()))
    end

    @testset "plate" begin
        @testset "replace_loop_idx_with_iters" begin
            f(expr) = Wren.replace_loop_idx_with_iters(expr, :i, :(1:J))
            @test f(:x) == :x
            @test f(:i) == :(1:J)
            @test f(:(x[i])) == :(x[1:J])
            @test f(:(x[i, j])) == :(x[1:J, j])
            @test f(:(x[j, i])) == :(x[j, 1:J])
            @test f(:(x[j, k])) == :(x[j, k])
            @test f(:(x.a)) == :(x.a)
            @test f(:(x.a[i])) == :(x.a[1:J])
            @test f(:(x.a[i, j])) == :(x.a[1:J, j])
            @test f(:(Normal(x[i]))) == :(Normal(x[1:J]))
            @test f(:(Normal(x[i, j]))) == :(Normal(x[1:J, j]))
            @test f(:(Normal(x.a[i]))) == :(Normal(x.a[1:J]))

            # Broadcasting operations where needed...
            @test f(:(i + 1)) == _normalise_ast!(:(@__dot__ (1:J) + 1))
            @test f(:(x[i+1])) == _normalise_ast!(:(x[@__dot__ (1:J) + 1]))
            @test f(:(x[-i])) == _normalise_ast!(:(x[@__dot__ -(1:J)]))

            @test_broken false
            # The following expression should error:
            # f(:(x[i].a))
        end
    end
end

end # module WrenExprTests
