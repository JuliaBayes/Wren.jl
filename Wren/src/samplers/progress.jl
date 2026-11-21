using Random: Random
using MCMCProgress: MCMCProgress
using ProgressLogging: ProgressLogging
using Term: Term

shows_progress(progress_backend) = progress_backend !== ()

function with_progress(f, progress_backend, label::AbstractString, nchains::Integer)
    shows_progress(progress_backend) || return f(nothing)
    # The refresh task must not change the chains' random streams.
    caller_rng = copy(Random.TaskLocalRNG())
    return MCMCProgress.progress(; label, nchains, backend=progress_backend) do run
        copy!(Random.TaskLocalRNG(), caller_rng)
        return f(run)
    end
end
