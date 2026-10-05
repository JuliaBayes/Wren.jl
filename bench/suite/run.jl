# Benchmarks each model with each backend and writes the results to a JSON file in
# `bench/suite/results/`, named `<UTC timestamp>-<short git sha>.json`. The file is then
# saved to `runs/` on the `benchmark-results` branch, and the branch is pushed.
#
#     julia --project=bench bench/suite/run.jl [--label <text>] [--no-save]
#
# The label is shown on the website to identify the run (e.g. "WrenCore v0.2.0" or
# "PR #123"). If not given, it defaults to the short git sha.
#
# With `--no-save`, the results are only written to `bench/suite/results/`. Runs with
# uncommitted changes can't be saved, since the git sha wouldn't describe the code that was
# benchmarked.
#
# The JSON files are read by the website in `bench/web`.

const USAGE = "Usage: julia --project=bench bench/suite/run.jl [--label <text>] [--no-save]"

# Check the arguments before loading anything, since that takes a while.
function parse_args(args)
    label = nothing
    save = true
    i = 1
    while i <= length(args)
        if args[i] == "--label" && i < length(args)
            label = args[i+1]
            i += 2
        elseif args[i] == "--no-save"
            save = false
            i += 1
        else
            println(stderr, USAGE)
            exit(1)
        end
    end
    return label, save
end
const LABEL, SAVE = parse_args(ARGS)

const REPO = readchomp(`git -C $(@__DIR__) rev-parse --show-toplevel`)
# Changes to files that git doesn't track (e.g. old results) don't count.
const DIRTY = !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`))
const RESULTS_BRANCH = "benchmark-results"

# Likewise, check that the results can be saved before spending time on the benchmarks.
if SAVE
    if DIRTY
        println(stderr, "There are uncommitted changes. Commit them, or use --no-save.")
        exit(1)
    end
    run(`git -C $REPO fetch origin $RESULTS_BRANCH`)
end

using WrenCore: benchmark_model, AutoEnzyme, AutoMooncake, Reverse
# Loaded so that WrenCore can use them as AD backends.
using Enzyme: Enzyme
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
    GLMPoisson,
    GLMMPoisson,
    GPRegr,
    KidscoreInteraction,
    Mh,
    Rate1,
    SblriBLR,
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
    git_sha = readchomp(`git -C $REPO rev-parse HEAD`)
    # When the benchmarked commit was made, in UTC. Unlike `timestamp` (when the benchmark
    # was run), this is still right if an old commit is benchmarked later.
    cmd = `git -C $REPO log -1 --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ`
    commit_date = readchomp(addenv(cmd, "TZ" => "UTC"))
    # The version of every package in the manifest, so that changes in dependencies can be
    # tracked across runs. Packages without a version (e.g. some stdlibs) are `nothing`.
    manifest = Dict(
        info.name => (info.version === nothing ? nothing : string(info.version)) for
        info in values(Pkg.dependencies())
    )
    short_sha = first(git_sha, 7)
    # Timings are only comparable between runs on the same kind of machine.
    machine = Dict(
        "cpu" => Sys.cpu_info()[1].model,
        "cpu_threads" => Sys.CPU_THREADS,
        "memory_gb" => round(Int, Sys.total_memory() / 2^30),
        "os" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
    )
    timestamp = now(UTC)
    output = Dict(
        "schema_version" => 1,
        "meta" => Dict(
            "timestamp" => string(timestamp) * "Z",
            "git_sha" => git_sha,
            "commit_date" => commit_date,
            "label" =>
                LABEL === nothing ? (DIRTY ? "$short_sha-dirty" : short_sha) : LABEL,
            # Whether there were uncommitted changes, i.e. whether `git_sha` describes the
            # code that was benchmarked.
            "dirty" => DIRTY,
            "julia_version" => string(VERSION),
            "machine" => machine,
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
    SAVE && save(outfile, short_sha)
    return nothing
end

# Copies `outfile` to `runs/` on the results branch, updates `runs/index.json`, and pushes
# the branch. This is done in a temporary worktree, so the current checkout isn't touched.
function save(outfile, short_sha)
    # Fetch again, in case the branch was pushed to while the benchmarks were running.
    run(`git -C $REPO fetch origin $RESULTS_BRANCH`)
    dir = mktempdir()
    run(`git -C $REPO worktree add --detach $dir origin/$RESULTS_BRANCH`)
    try
        runs = joinpath(dir, "runs")
        mkpath(runs)
        cp(outfile, joinpath(runs, basename(outfile)))
        write_index(runs)
        run(`git -C $dir add runs`)
        run(`git -C $dir commit -m "Add benchmark results for $short_sha"`)
        run(`git -C $dir push origin HEAD:$RESULTS_BRANCH`)
    finally
        run(`git -C $REPO worktree remove --force $dir`)
    end
    println("Results saved to the $RESULTS_BRANCH branch")
    return nothing
end

# Writes `<runs>/index.json`, which lists every run in `runs`, newest first. The website
# reads it to find the runs.
function write_index(runs)
    entries = Dict{String,Any}[]
    for file in readdir(runs)
        (endswith(file, ".json") && file != "index.json") || continue
        meta = JSON.parsefile(joinpath(runs, file))["meta"]
        push!(
            entries,
            Dict(
                "file" => file,
                "timestamp" => meta["timestamp"],
                "git_sha" => get(meta, "git_sha", nothing),
                "commit_date" => get(meta, "commit_date", nothing),
                "label" => get(meta, "label", nothing),
            ),
        )
    end
    sort!(entries; by=e -> e["timestamp"], rev=true)
    open(io -> JSON.json(io, entries; pretty=true), joinpath(runs, "index.json"), "w")
    return nothing
end

main()
