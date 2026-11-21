using Random: Random
using MCMCProgress: MCMCProgress, TermBackend, ProgressLoggingBackend, PlainTextBackend
using ProgressLogging: ProgressLogging
using Term: Term

# TODO: replace with `MCMCProgress.AbstractBackend` once MCMCProgress v0.2 is released
const ProgressBackend = Union{TermBackend,ProgressLoggingBackend,PlainTextBackend}
const ProgressSpec = Union{Bool,ProgressBackend,Tuple{Vararg{ProgressBackend}}}

progress_backends(show_progress::Bool) = show_progress ? (TermBackend(),) : ()
progress_backends(backend::ProgressBackend) = (backend,)
progress_backends(backends::Tuple{Vararg{ProgressBackend}}) = backends

function with_progress(f, backends::Tuple, label::AbstractString, nchains::Integer)
    isempty(backends) && return f(nothing)
    # The refresh task must not change the chains' random streams.
    caller_rng = copy(Random.TaskLocalRNG())
    return MCMCProgress.progress(; label, nchains, backend=backends) do run
        copy!(Random.TaskLocalRNG(), caller_rng)
        return f(run)
    end
end
