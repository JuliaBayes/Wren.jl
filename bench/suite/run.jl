# Benchmarks each model with each backend and writes the results to a JSON file in
# `bench/suite/results/`, named `<UTC timestamp>-<short git sha>.json`.
#
#     julia --project=bench bench/suite/run.jl [--label <text>]
#
# The label is shown on the website to identify the run (e.g. "WrenCore v0.2.0" or
# "PR #123"). If not given, it defaults to the short git sha.
#
# The JSON files are read by the website in `bench/web`.

# Check the arguments before loading anything, since that takes a while.
const LABEL = if isempty(ARGS)
    nothing
elseif length(ARGS) == 2 && ARGS[1] == "--label"
    ARGS[2]
else
    println(stderr, "Usage: julia --project=bench bench/suite/run.jl [--label <text>]")
    exit(1)
end

using Wren: benchmark_model
using ADTypes: AutoEnzyme, AutoMooncake
using Enzyme: Reverse
using Mooncake: Mooncake
using JSON: JSON
using Dates: Dates, now, UTC, @dateformat_str
using Pkg: Pkg

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

# Each module must define `wren_model`. The module names are used to match up models
# across runs, so don't change them.
const MODELS = (
    ArK,
    EightSchoolsCentered,
    GLMBinomial,
    # GLMPoisson,
    # GLMMPoisson,
    # GPRegr,
    # KidscoreInteraction,
    # Mh,
    # Rate1,
    # SblriBLR,
)

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

function main()
    results = Dict{String,Any}()
    for mod in MODELS
        name = string(nameof(mod))
        model = mod.wren_model
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
    # The version of every package in the manifest, so that changes in dependencies can be
    # tracked across runs. Packages without a version (e.g. some stdlibs) are `nothing`.
    manifest = Dict(
        info.name => (info.version === nothing ? nothing : string(info.version)) for
        info in values(Pkg.dependencies())
    )
    short_sha = git_sha === nothing ? "unknown" : first(git_sha, 7)
    timestamp = now(UTC)
    output = Dict(
        "schema_version" => 1,
        "meta" => Dict(
            "timestamp" => string(timestamp) * "Z",
            "git_sha" => git_sha,
            "label" => LABEL === nothing ? short_sha : LABEL,
            "julia_version" => string(VERSION),
        ),
        # Lists the backends in order, since the keys in `results` get sorted.
        "backends" => first.(BACKENDS),
        "results" => results,
        "manifest" => manifest,
    )
    # The timestamp comes first so that the files sort by time.
    filename =
        Dates.format(timestamp, dateformat"yyyymmdd\THHMMSS\Z") * "-" * short_sha * ".json"
    outfile = joinpath(@__DIR__, "results", filename)
    mkpath(dirname(outfile))
    open(io -> JSON.json(io, output; pretty=true), outfile, "w")
    println("Results written to $outfile")
    return nothing
end

main()
