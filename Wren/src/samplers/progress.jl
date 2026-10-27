using Random: Random
using MCMCProgress: MCMCProgress, TermBackend
# MCMCProgress renders through Term only once Term itself is loaded, which is
# what this import is for.
using Term: Term

"""
    BACKEND

The display Wren's samplers draw into: MCMCProgress.jl's Term backend, which
gives every phase of every chain a bar of its own.
"""
const BACKEND = TermBackend()

"""
    withprogress(f, show_progress::Bool, label::AbstractString, nchains::Integer)

Call `f(run)` while the progress of `nchains` chains is displayed under `label`,
and return whatever `f` returns. `run` is the `MCMCProgress` run the chains
report into.

With `show_progress` false nothing is displayed and `f` is called with `nothing`
in place of a run, which is the signal to its callees to skip reporting
altogether.

`f` must wait for every task it spawns before returning.

Spawning a task changes the random streams of the caller's later spawns, and
MCMCProgress spawns a refresh task before calling `f`. The caller's task-local
RNG is therefore restored before `f` runs, so tasks spawned by `f` draw the
same random numbers whether or not progress is shown.
"""
function withprogress(f, show_progress::Bool, label::AbstractString, nchains::Integer)
    show_progress || return f(nothing)
    caller_rng = copy(Random.TaskLocalRNG())
    return MCMCProgress.progress(; label, nchains, backend=BACKEND) do run
        copy!(Random.TaskLocalRNG(), caller_rng)
        return f(run)
    end
end
