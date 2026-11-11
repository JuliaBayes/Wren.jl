using Random: Random
using MCMCProgress: MCMCProgress, TermBackend
using Term: Term

const BACKEND = TermBackend()

function withprogress(f, show_progress::Bool, label::AbstractString, nchains::Integer)
    show_progress || return f(nothing)
    # The refresh task must not change the chains' random streams.
    caller_rng = copy(Random.TaskLocalRNG())
    return MCMCProgress.progress(; label, nchains, backend=BACKEND) do run
        copy!(Random.TaskLocalRNG(), caller_rng)
        return f(run)
    end
end
