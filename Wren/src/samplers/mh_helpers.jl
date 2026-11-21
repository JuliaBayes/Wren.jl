using LinearAlgebra: Cholesky, Diagonal, lmul!
using MCMCProgress: Determinate, advance!, chain_at, close_phase!, open_phase!
using Random: Random
using VarNames: VarName

function _check_mh_arguments(niters, nchains, max_init_attempts)
    niters > 0 || throw(ArgumentError("niters must be positive"))
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    max_init_attempts > 0 || throw(ArgumentError("max_init_attempts must be positive"))
    return nothing
end

function _mh_start(draw, max_init_attempts, quantity)
    for i in 1:max_init_attempts
        state, density = draw(i)
        isfinite(density) && return state, density
    end
    throw(ArgumentError("No finite $quantity after $max_init_attempts prior draws"))
end

function _mh_phase(f, run, j, label, n, args...)
    run === nothing && return f(nothing, args...)
    phase = open_phase!(chain_at(run, j), label, Determinate(n))
    try
        return f(phase, args...)
    finally
        close_phase!(phase)
    end
end

function _device_mh_phase(backend, step, state, history, n, run, j, label)
    scalar = backend._device_scalar
    return _mh_phase(run, j, label, n, state, history) do phase, state, history
        batch_size = phase === nothing ? n : min(n, max(32, cld(n, 100)))
        scan = backend._device_compile(
            backend._device_scan,
            step,
            state,
            history,
            scalar(batch_size),
            scalar(0),
        )
        for offset in 0:batch_size:(n-1)
            count = min(batch_size, n - offset)
            state, history = scan(step, state, history, scalar(count), scalar(offset))
            phase === nothing && continue
            backend._device_wait(state)
            advance!(phase, offset + count)
        end
        return state, history
    end
end

function _run_mh_chains(f, rng, model, nchains, progress)
    seeds = rand(rng, UInt64, nchains)
    outputs = with_progress(progress, "Sampling $(model.name)", nchains) do run
        tasks = @sync [Threads.@spawn f(Random.Xoshiro(seeds[j]), run, j) for j in 1:nchains]
        fetch.(tasks)
    end
    return stack(first.(outputs)), stack(last.(outputs))
end

function _mh_backend(with_reactant::Bool)
    with_reactant || return nothing
    backend = Base.get_extension(WrenCore, :WrenCoreReactantExt)
    backend === nothing &&
        throw(ArgumentError("Load Reactant and Enzyme before setting with_reactant=true"))
    return backend
end

function _random_walk(algorithm, model::WrenCore.Model, args...; kwargs...)
    return _random_walk(algorithm, Random.default_rng(), model, args...; kwargs...)
end

function _random_walk(
    algorithm,
    rng::Random.AbstractRNG,
    model::WrenCore.Model,
    niters::Integer,
    nchains::Integer=1;
    step_size::Union{Nothing,Real}=nothing,
    warmup::Integer=0,
    progress::Bool=true,
    track::TrackedQuantitySpec=AllQuantities(),
    max_init_attempts::Integer=1000,
    with_reactant::Bool=false,
)
    _check_mh_arguments(niters, nchains, max_init_attempts)
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative"))
    if step_size !== nothing && !(0 < step_size < Inf)
        throw(ArgumentError("step_size must be finite and positive"))
    end
    backend = _mh_backend(with_reactant)
    options = (; warmup, step_size, track, max_init_attempts)
    results, stats = _run_mh_chains(rng, model, nchains, progress) do chain_rng, run, j
        source = _ContinuousPrior(WrenCore.FromPrior(chain_rng))
        vm = WrenCore.VecModel(
            WrenCore._copy_for_thread(model),
            nothing;
            latent_source=source,
        )
        vm.dimension > 0 ||
            throw(ArgumentError("Random-walk samplers require latent variables"))
        _walk_chain(backend, algorithm, source, vm, niters, run, j, options)
    end
    return WrenCore.to_flexichain(results, stats)
end

struct _ContinuousPrior{S<:WrenCore.FromPrior} <: WrenCore.AbstractLatentSource
    source::S
end

WrenCore.scope(source::_ContinuousPrior, ::VarName) = source

function WrenCore.get_latent(source::_ContinuousPrior, vn::VarName, dist, site::Val)
    continuous = WrenCore.Plaice.is_continuous
    if applicable(continuous, dist) && !continuous(dist)
        throw(ArgumentError("Random-walk samplers require continuous latents. Got `$vn`."))
    end
    return WrenCore.get_latent(source.source, vn, dist, site)
end

function _walk_chain(::Nothing, algorithm, source, vm, niters, run, j, options)
    (; warmup, step_size, track, max_init_attempts) = options
    rng = source.source.rng
    log_density = Base.Fix1(WrenCore.lp, vm)
    x, density = _random_walk_start(vm, source, max_init_attempts) do candidate
        return candidate, _random_walk_density(log_density, candidate)
    end
    x = copy(x)
    step, scale, variance = _walk_scales(algorithm, x, step_size)
    factor = _random_walk_factor(algorithm, step, length(x))
    adaptation = warmup == 0 ? nothing : _walk_adaptation(algorithm, x, scale, variance)
    proposal = similar(x)
    noise = (; first=similar(x), retry=algorithm isa _DRAM ? similar(x) : nothing)
    results = Vector{WrenCore.ModelResult}(undef, niters)
    stats = Vector{NamedTuple}(undef, niters)

    result = nothing

    for (label, n, adapting) in (("Warmup", warmup, true), ("Sampling", niters, false))
        n == 0 && continue
        phase =
            run === nothing ? nothing : open_phase!(chain_at(run, j), label, Determinate(n))
        try
            for i in 1:n
                next_density, stage, log_alpha = _walk_step!(
                    algorithm,
                    rng,
                    log_density,
                    x,
                    density,
                    _proposal_factor(factor),
                    proposal,
                    noise,
                )
                if stage != 0
                    x, proposal = proposal, x
                    density = next_density
                end
                if adapting
                    factor = _adapt_walk!(
                        adaptation,
                        factor,
                        x,
                        noise.first,
                        log_alpha,
                        i,
                        i == n,
                    )
                else
                    if i == 1 || stage != 0
                        # Saved outputs may borrow the vector that holds the latents.
                        result = WrenCore.simulate(vm, copy(x); track)
                    end
                    results[i] = result
                    stats[i] = _random_walk_stats(algorithm, density, stage)
                end
                phase === nothing || advance!(phase, i)
            end
        finally
            phase === nothing || close_phase!(phase)
        end
    end
    return results, stats
end

function _walk_scales(algorithm, x, step_size)
    T = eltype(x)
    scale = T(2.38) / sqrt(T(length(x)))
    step = step_size === nothing ? scale : T(step_size)
    variance = abs2(step)
    0 < variance < Inf || throw(ArgumentError("step_size is out of range for $T"))
    if algorithm isa _RAM && !(0 < T(algorithm.target_acceptance) < 1)
        throw(ArgumentError("target_acceptance rounds to an endpoint in $T"))
    end
    if algorithm isa _DRAM && !(0 < T(algorithm.second_scale) < 1)
        throw(ArgumentError("second_scale rounds to an endpoint in $T"))
    end
    return step, scale, variance
end

_random_walk_factor(algorithm, step, d) = Cholesky(Matrix(Diagonal(fill(step, d))), 'L', 0)

_proposal_factor(factor::Diagonal) = factor
_proposal_factor(factor::Cholesky) = factor.L

function _random_walk_start(score, vm, source, max_init_attempts)
    return _mh_start(max_init_attempts, "log density") do i
        candidate = i == 1 ? vm.sample : WrenCore.vector_parameters(vm, source)
        return score(candidate)
    end
end

function _random_walk_density(log_density, x)
    all(isfinite, x) || return -Inf
    density = log_density(x)
    density < Inf || throw(DomainError(density, "Expected finite log density or -Inf"))
    return density
end

function _random_walk_proposal!(proposal, x, L, noise)
    copyto!(proposal, noise)
    lmul!(L, proposal)
    proposal .+= x
    return proposal
end

function _walk_step!(algorithm, rng, log_density, x, density, L, proposal, noise)
    Random.randn!(rng, noise.first)
    _random_walk_proposal!(proposal, x, L, noise.first)
    first = _random_walk_density(log_density, proposal)
    log_alpha = min(0, first - density)
    log(rand(rng)) < log_alpha && return first, 1, log_alpha
    if algorithm isa _DRAM
        Random.randn!(rng, noise.retry)
        noise.retry .*= algorithm.second_scale
        _random_walk_proposal!(proposal, x, L, noise.retry)
        second = _random_walk_density(log_density, proposal)
        log_alpha2 = _delayed_log_acceptance(density, first, second, log_alpha, noise)
        log(rand(rng)) < log_alpha2 && return second, 2, log_alpha
    end
    return density, 0, log_alpha
end

_random_walk_stats(algorithm, log_density, stage) = (; accepted=stage != 0, log_density)

_adapt_walk!(::Nothing, factor, x, noise, log_alpha, i, last) = factor
