# Benchmarks each model with each backend and writes the results to a JSON file in
# `bench/suite/results/`, named `<UTC timestamp>-<short git sha>.json`. The file is then
# saved to `runs/` on the `benchmark-results` branch, and the branch is pushed.
#
#     julia --project=bench bench/suite/run.jl [--label <text> | --release] [--no-save]
#
# The label is shown on the website to identify the run (e.g. "WrenCore-v0.2.0" or
# "PR #123"). If not given, it defaults to the subject line of the commit's message.
#
# `--release` labels the run with the release's tag, `WrenCore-v<version>`, using the
# version in `WrenCore/Project.toml`. The current commit must have that tag.
#
# With `--no-save`, the results are only written to `bench/suite/results/`. Runs with
# uncommitted changes can't be saved, since the git sha wouldn't describe the code that was
# benchmarked.
#
# The JSON files are read by the website in `bench/web`.

const USAGE = "Usage: julia --project=bench bench/suite/run.jl [--label <text> | --release] [--no-save]"

# Check the arguments before loading anything, since that takes a while.
function parse_args(args)
    label = nothing
    release = false
    save = true
    i = 1
    while i <= length(args)
        if args[i] == "--label" && i < length(args)
            label = args[i+1]
            i += 2
        elseif args[i] == "--release"
            release = true
            i += 1
        elseif args[i] == "--no-save"
            save = false
            i += 1
        else
            println(stderr, USAGE)
            exit(1)
        end
    end
    if release && label !== nothing
        println(stderr, "--label and --release can't be used together.")
        exit(1)
    end
    return label, release, save
end
const ARG_LABEL, RELEASE, SAVE = parse_args(ARGS)

using TOML: TOML

# Defines `REPO` and `RESULTS_BRANCH`, and the code for saving to the results branch.
include("results_branch.jl")

# The label for a release run, which is the release's tag, e.g. "WrenCore-v0.2.0". Checks
# that the current commit has that tag (which TagBot creates when a release is made).
function release_label()
    project = joinpath(REPO, "WrenCore", "Project.toml")
    tag = "WrenCore-v" * TOML.parsefile(project)["version"]
    if !(tag in split(readchomp(`git -C $REPO tag --points-at HEAD`)))
        println(
            stderr,
            "The current commit doesn't have the tag $tag. Check out that tag (you may " *
            "need to run `git fetch --tags` first).",
        )
        exit(1)
    end
    return tag
end

# Changes to files that git doesn't track (e.g. old results) don't count.
const DIRTY = !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`))

const LABEL = if RELEASE
    release_label()
elseif ARG_LABEL !== nothing
    ARG_LABEL
else
    # The website already shows the sha next to the label, so use the commit message as the
    # default label rather than the sha.
    subject = readchomp(`git -C $REPO log -1 --format=%s`)
    DIRTY ? "$subject (dirty)" : subject
end

# Likewise, check that the results can be saved before spending time on the benchmarks.
if SAVE
    if DIRTY
        println(stderr, "There are uncommitted changes. Commit them, or use --no-save.")
        exit(1)
    end
    fetch_results_branch()
end

# BLAS's own threads make small matrix operations (as in GPRegr) much slower in some Julia
# sessions but not others, e.g. a gradient that usually takes 5 μs taking 5–460 μs. One
# BLAS thread makes the results consistent, and the matrices here are too small to benefit
# from more.
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

yesno(x::Bool) = x ? "Yes" : "No"
println("Commit: ", readchomp(`git -C $REPO rev-parse --short HEAD`))
println("Label: ", LABEL)
println("Is release: ", yesno(RELEASE))
println("Save to $RESULTS_BRANCH: ", yesno(SAVE))
println("Julia threads: ", Threads.nthreads())
println()

using WrenCore: benchmark_model, posteriordb_model, AutoEnzyme, AutoMooncake, Reverse
# Loaded so that WrenCore can build the PosteriorDB models.
using PosteriorDB: PosteriorDB
using WrenDists: WrenDists
# Loaded so that WrenCore can use them as AD backends.
using Enzyme: Enzyme
using Mooncake: Mooncake
using JSON: JSON
using Dates: Dates, now, UTC, @dateformat_str
using Pkg: Pkg
using Printf: @sprintf

# `nothing` means the primal (no AD). Add more backends here.
const BACKENDS = [
    "primal" => nothing,
    "enzyme_reverse" => AutoEnzyme(; mode=Reverse),
    "mooncake_reverse" => AutoMooncake(; config=nothing),
]

# Maps the names of the models to their PosteriorDB posterior names. The names are used to
# match up models across runs, so don't change them. The website links each name to the
# model's file, using `MODEL_FILES` in `bench/web/src/lib/github.ts`; update that when
# adding a model.
const MODELS = (
    "ArK" => "arK-arK",
    "EightSchoolsCentered" => "eight_schools-eight_schools_centered",
    "GLMBinomial" => "GLM_Binomial_data-GLM_Binomial_model",
    "GLMPoisson" => "GLM_Poisson_Data-GLM_Poisson_model",
    "GLMMPoisson" => "GLMM_Poisson_data-GLMM_Poisson_model",
    "GPRegr" => "gp_pois_regr-gp_regr",
    "KidscoreInteraction" => "kidiq-kidscore_interaction",
    "Mh" => "Mh_data-Mh_model",
    "Rate1" => "Rate_1_data-Rate_1_model",
    "SblriBLR" => "sblri-blr",
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

# Anything beyond 3 significant figures is just noise.
const SIGFIGS = 3
# `%g` drops trailing zeros, so that e.g. 120.0 is shown as 120.
sigfigs(x::Real) = @sprintf("%g", round(x; sigdigits=SIGFIGS))

# Formats a time in seconds, e.g. `1.23 μs`. This matches `formatTime` in
# `bench/web/src/lib/format.ts`.
function format_time(x::Real)
    # Round before picking the unit, so that e.g. 999.7 ns is shown as 1 μs, not 1000 ns.
    x = round(x; sigdigits=SIGFIGS)
    return if x < 1e-6
        "$(sigfigs(x * 1e9)) ns"
    elseif x < 1e-3
        "$(sigfigs(x * 1e6)) μs"
    elseif x < 1
        "$(sigfigs(x * 1e3)) ms"
    else
        "$(sigfigs(x)) s"
    end
end

# Prints the results as a table, with one row per model and one column per backend. Each
# AD backend's time is followed by its ratio to the primal time. Errors are shown in the
# table as `ERR [k]`, and listed in full after the table.
function print_results(io::IO, results)
    backends = first.(BACKENDS)
    errors = Tuple{String,String,String}[]
    header = ["Model"; "dim"; backends]
    rows = map(collect(first.(MODELS))) do name
        r = results[name]
        primal = get(r["primal"], "time", nothing)
        cells = map(backends) do backend
            entry = r[backend]
            if haskey(entry, "error")
                push!(errors, (name, backend, entry["error"]))
                return "ERR [$(length(errors))]"
            end
            time = entry["time"]
            # No ratio for the primal itself, or if the primal failed.
            ratio =
                backend == "primal" || primal === nothing ? "" :
                " ($(sigfigs(time / primal))x)"
            return format_time(time) * ratio
        end
        [name; string(get(r, "dim", "?")); cells]
    end
    widths = [
        maximum(textwidth, [header[j]; [row[j] for row in rows]]) for j in eachindex(header)
    ]
    # Model names are left-aligned, everything else right-aligned.
    pad(s, j) = j == 1 ? rpad(s, widths[j]) : lpad(s, widths[j])
    line(cells) = "| " * join((pad(c, j) for (j, c) in enumerate(cells)), " | ") * " |"
    println(io)
    println(io, "Ratios are vs primal:")
    println(io, line(header))
    println(io, "|" * join(("-"^(w + 2) for w in widths), "|") * "|")
    foreach(row -> println(io, line(row)), rows)
    println(io)
    if !isempty(errors)
        println(io, "Errors:")
        for (k, (name, backend, msg)) in enumerate(errors)
            msg = join(split(msg, "\n"), "; ")
            # Truncate long messages
            MAX_LEN = 200
            if length(msg) > MAX_LEN
                msg = first(msg, MAX_LEN) * "…"
            end
            println(io, "  [$k] $name, $backend: $msg")
        end
        println(io)
    end
    return nothing
end

function main()
    results = Dict{String,Any}()
    N = length(MODELS)
    for (i, (name, pdb_name)) in enumerate(MODELS)
        model = posteriordb_model(pdb_name)
        model_results = Dict{String,Any}()
        for (backend, adtype) in BACKENDS
            println("(model $i/$N) Benchmarking $name with $backend...")
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
        # Timings also depend on these settings.
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
    )
    timestamp = now(UTC)
    output = Dict(
        "schema_version" => 1,
        "meta" => Dict(
            "timestamp" => string(timestamp) * "Z",
            "git_sha" => git_sha,
            "commit_date" => commit_date,
            "label" => LABEL,
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
    print_results(stdout, results)
    println("Results written to $outfile")
    if SAVE
        message = "Add benchmark results for $short_sha"
        update_results_branch(
            runs -> cp(outfile, joinpath(runs, basename(outfile))),
            message,
        )
        println("Results saved to the $RESULTS_BRANCH branch")
    end
    return nothing
end

main()
