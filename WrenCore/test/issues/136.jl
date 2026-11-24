module WrenIssue136Tests

using Test
using WrenCore
using Logging

const HAS_FUNCTION_MACRO = isdefined(Base, Symbol("@__FUNCTION__"))
const THIS_FILE = "WrenCore/test/issues/136.jl"

raise_issue_error(x) = WrenCore.@_issue_error("something went wrong with $x")

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

    @testset "dataflow debug logging" begin
        model_expr = :(@wren function debug_model(a)
            x ~ Normal(a)
        end)

        logs, _ = Test.collect_test_logs(min_level=Logging.Debug) do
            macroexpand(@__MODULE__, model_expr)
        end

        record = only(filter(r -> r.message == "Dataflow analysis", logs))

        @test record.level == Logging.Debug
        @test endswith(replace(string(record.file), '\\' => '/'), "model/dataflow.jl")
        @test all(
            k -> haskey(record.kwargs, k),
            (:lhs_syms, :required_syms, :tq_uses, :args_to_copy),
        )
    end
end

end # module WrenIssue136Tests
