using Random: Random
using MCMCProgress: MCMCProgress, TermBackend, ProgressLoggingBackend, PlainTextBackend, AbstractBackend
using ProgressLogging: ProgressLogging
using Term: Term

const ProgressSpec = Union{Bool,AbstractBackend,Tuple{Vararg{AbstractBackend}}}

progress_backends(show_progress::Bool) = show_progress ? (TermBackend(),) : ()
progress_backends(backend::AbstractBackend) = (backend,)
progress_backends(backends::Tuple{Vararg{AbstractBackend}}) = backends

function with_progress(f, backends::Tuple, label::AbstractString, nchains::Integer)
    isempty(backends) && return f(nothing)
    # The refresh task must not change the chains' random streams.
    caller_rng = copy(Random.TaskLocalRNG())
    return MCMCProgress.progress(; label, nchains, backend=backends) do run
        copy!(Random.TaskLocalRNG(), caller_rng)
        return f(run)
    end
end
