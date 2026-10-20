# Removes runs from `runs/` on the `benchmark-results` branch, updates `runs/index.json`,
# and pushes the branch.
#
#     julia --project=bench bench/suite/remove.jl <file or sha> [<file or sha> ...]
#
# Each argument is either the name of a run's file (e.g. `20261005T205315Z-264e448.json`),
# or a git sha (at least 7 characters), which matches every run of that commit. The matching
# runs are listed, and you're asked to confirm before anything is removed.
#
# Removed runs are still in the branch's history, so they can be brought back if needed.

if isempty(ARGS)
    println(stderr, "Usage: julia --project=bench bench/suite/remove.jl <file or sha> ...")
    exit(1)
end

include("results_branch.jl")

function find_runs(args)
    fetch_results_branch()
    index =
        JSON.parse(readchomp(`git -C $REPO show origin/$RESULTS_BRANCH:runs/index.json`))
    println("Runs to remove from the $RESULTS_BRANCH branch:")
    matches = String[]
    for arg in args
        found = filter(index) do entry
            sha = something(entry["git_sha"], "")
            entry["file"] == arg || (length(arg) >= 7 && startswith(sha, arg))
        end
        if isempty(found)
            println(stderr, "No runs match $arg.")
            exit(1)
        end
        for entry in found
            entry["file"] in matches && continue
            push!(matches, entry["file"])
            label = something(entry["label"], "")
            println("  ", entry["file"], "  (", label, ")")
        end
    end
    return matches
end

function main()
    files = find_runs(ARGS)
    print("Remove $(length(files)) run(s)? [y/N] ")
    if lowercase(strip(readline())) != "y"
        println("Nothing was removed.")
        return nothing
    end
    message = "Remove benchmark results for $(join(ARGS, ", "))"
    update_results_branch(message) do runs
        foreach(file -> rm(joinpath(runs, file)), files)
    end
    println("Removed $(length(files)) run(s) from the $RESULTS_BRANCH branch")
    return nothing
end

main()
