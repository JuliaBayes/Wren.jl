# Benchmarks each model with each backend and writes the results to a JSON file.
#
#     julia --project=bench bench/suite/run.jl [output.json]
#
# The JSON files are read by the website in `bench/web`.

using Wren: benchmark_model
using ADTypes: AutoEnzyme, AutoMooncake
using Enzyme: Reverse
using Mooncake: Mooncake
using JSON: JSON
using Dates: now, UTC

# `nothing` means the primal (no AD). Add more backends here.
const BACKENDS = [
    "primal" => nothing,
    "enzyme_reverse" => AutoEnzyme(; mode=Reverse),
    "mooncake_reverse" => AutoMooncake(; config=nothing),
]

include("models/ark.jl")
include("models/eight_schools_centered.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/kidscore_interaction.jl")
include("models/mh.jl")
include("models/rate1.jl")
include("models/sblri_blr.jl")

# The names are used to match up models across runs, so don't change them.
const MODELS = [
    "ark" => ArK.wren_model,
    "eight_schools_centered" => EightSchoolsCentered.wren_model,
    "glm_binomial" => GLMBinomial.wren_model,
    "glm_poisson" => GLMPoisson.wren_model,
    "glmm_poisson" => GLMMPoisson.wren_model,
    "gp_regr" => GPRegr.wren_model,
    "kidscore_interaction" => KidscoreInteraction.wren_model,
    "mh" => Mh.wren_model,
    "rate1" => Rate1.wren_model,
    "sblri_blr" => SblriBLR.wren_model,
]

function bench(model, adtype)
    try
        r = benchmark_model(model, adtype; verbose=false)
        sample = adtype === nothing ? r.lp_bench : r.lp_and_grad_bench
        return Dict("time" => sample.time, "dim" => length(r.vecmodel.sample))
    catch e
        msg = sprint(showerror, e)
        return Dict("error" => first(msg, 500))
    end
end

function main(outfile)
    results = Dict{String,Any}()
    for (name, model) in MODELS
        model_results = Dict{String,Any}()
        for (backend, adtype) in BACKENDS
            println("Benchmarking $name with $backend...")
            r = bench(model, adtype)
            # The dimension is the same for every backend, so store it once per model.
            haskey(r, "dim") && (model_results["dim"] = pop!(r, "dim"))
            model_results[backend] = r
        end
        results[name] = model_results
    end
    git_sha = try
        readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
    catch
        nothing
    end
    output = Dict(
        "schema_version" => 1,
        "meta" => Dict(
            "timestamp" => string(now(UTC)) * "Z",
            "git_sha" => git_sha,
            "julia_version" => string(VERSION),
        ),
        "results" => results,
    )
    mkpath(dirname(abspath(outfile)))
    open(io -> JSON.json(io, output; pretty=true), outfile, "w")
    println("Results written to $outfile")
    return nothing
end

main(isempty(ARGS) ? "results.json" : ARGS[1])
