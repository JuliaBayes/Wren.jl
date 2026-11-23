module WrenIssue136Tests

using Test
using WrenCore

const HAS_FUNCTION_MACRO = isdefined(Base, Symbol("@__FUNCTION__"))
const THIS_FILE = "WrenCore/test/issues/136.jl"

raise_issue_error(x) = WrenCore.@_issue_error("something went wrong with $x")
log_debug(x) = WrenCore.@_compilation_debug("a message", x, y=2x)

function with_debug(f)
    old = WrenCore.WREN_COMPILATION_DEBUG[]
    WrenCore.WREN_COMPILATION_DEBUG[] = true
    try
        return f()
    finally
        WrenCore.WREN_COMPILATION_DEBUG[] = old
    end
end

@testset "source locations in internal errors and debug logs" begin
    @testset "@_issue_error" begin
        err = try
            raise_issue_error(1)
        catch e
            e
        end
        @test err isa ErrorException
        @test startswith(err.msg, "something went wrong with 1.")
        @test occursin("$THIS_FILE:", err.msg)
        @test occursin("https://github.com/JuliaBayes/Wren.jl/issues", err.msg)
        if HAS_FUNCTION_MACRO
            @test occursin("raise_issue_error at $THIS_FILE:", err.msg)
        end

        # An internal call site is shown relative to the package.
        err = try
            WrenCore.Expressions.get_top_level_symbol(:(f(x)))
        catch e
            e
        end
        @test occursin("WrenCore/src/model/expr.jl:", err.msg)
        if HAS_FUNCTION_MACRO
            @test occursin("get_top_level_symbol at WrenCore/src/model/expr.jl:", err.msg)
        end
    end

    @testset "@_compilation_debug" begin
        @test !WrenCore.WREN_COMPILATION_DEBUG[]
        @test_logs log_debug(1)

        logs, _ = Test.collect_test_logs() do
            with_debug(() -> log_debug(1))
        end
        record = only(logs)
        @test record.level == Base.CoreLogging.Info
        location = HAS_FUNCTION_MACRO ? "log_debug at $THIS_FILE" : THIS_FILE
        @test startswith(record.message, "[$location:")
        @test endswith(record.message, "] a message")
        @test record.kwargs == Dict(:x => 1, :y => 2)
        @test endswith(replace(record.file, '\\' => '/'), THIS_FILE)

        # The dataflow analysis, which runs when `@wren` is expanded, logs its results.
        model_expr = :(@wren function debug_model(a)
            x ~ Normal(a)
        end)
        expected =
            r"WrenCore/src/model/dataflow\.jl:\d+\] dataflow analysis\nlhs_syms = Set\("
        @test_logs (:info, expected) match_mode = :any with_debug(
            () -> macroexpand(@__MODULE__, model_expr),
        )
    end
end

end # module WrenIssue136Tests
