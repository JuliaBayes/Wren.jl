# Code for changing the `benchmark-results` branch, shared by `run.jl` and `remove.jl`.

using JSON: JSON

const REPO = readchomp(`git -C $(@__DIR__) rev-parse --show-toplevel`)
const RESULTS_BRANCH = "benchmark-results"

fetch_results_branch() = run(`git -C $REPO fetch origin $RESULTS_BRANCH`)

# Checks out the results branch in a temporary worktree, so the current checkout isn't
# touched. Then calls `f(runs)`, where `runs` is the path to the branch's `runs/` folder,
# updates `runs/index.json`, commits the changes with the given message, and pushes the
# branch.
function update_results_branch(f, message)
    # Fetch again, in case the branch was pushed to since it was last fetched.
    fetch_results_branch()
    dir = mktempdir()
    run(`git -C $REPO worktree add --detach $dir origin/$RESULTS_BRANCH`)
    try
        runs = joinpath(dir, "runs")
        mkpath(runs)
        f(runs)
        write_index(runs)
        # `--all` so that removed files are included.
        run(`git -C $dir add --all runs`)
        # The results branch has no pre-commit config, so tell pre-commit (if it's
        # installed as a git hook) not to fail because of that.
        commit = `git -C $dir commit -m $message`
        run(addenv(commit, "PRE_COMMIT_ALLOW_NO_CONFIG" => "1"))
        run(`git -C $dir push origin HEAD:$RESULTS_BRANCH`)
    finally
        run(`git -C $REPO worktree remove --force $dir`)
    end
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
