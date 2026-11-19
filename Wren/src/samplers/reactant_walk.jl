function _device_walk_chain(backend, algorithm, source, vm, niters, run, j, options)
    (; warmup, step_size, track, max_init_attempts) = options
    log_density = backend._device_log_density(vm)
    x, density, initial, density_type =
        _device_walk_start(backend, vm, source, max_init_attempts, log_density)
    step, scale, variance = _walk_scales(algorithm, vm.sample, step_size)
    factor = backend._device_array(_device_walk_factor(algorithm, step, vm.dimension))
    rng = backend._device_rng(source.source.rng)
    invalid = backend._device_scalar(false)

    if warmup > 0
        adaptation = _device_walk_adaptation(backend, algorithm, initial, scale, variance)
        state = (; rng, x, density, factor, adaptation, invalid)
        step_warmup =
            (state, i) ->
                _device_walk_warmup(backend, algorithm, log_density, state, i, warmup)
        state, _ =
            _device_mh_phase(backend, step_warmup, state, nothing, warmup, run, j, "Warmup")
        (; rng, x, density, factor, invalid) = state
    end

    history = map(
        backend._device_array,
        (
            zeros(eltype(vm.sample), vm.dimension, niters),
            zeros(density_type, niters),
            zeros(Int, niters),
        ),
    )
    state = (; rng, x, density, factor, invalid)
    step_sample = (state, _) -> _device_walk_sample(backend, algorithm, log_density, state)
    final, history =
        _device_mh_phase(backend, step_sample, state, history, niters, run, j, "Sampling")
    backend._to_host(final.invalid) && throw(
        DomainError(NaN, "Expected finite log density or -Inf and finite proposal factor"),
    )
    positions, scores, stages = map(Array, history)

    results = Vector{WrenCore.ModelResult}(undef, niters)
    stats = Vector{NamedTuple}(undef, niters)
    for i in 1:niters
        if i == 1 || stages[i] != 0
            results[i] = WrenCore.simulate(vm, positions[:, i]; track)
        else
            results[i] = results[i-1]
        end
        stats[i] = _random_walk_stats(algorithm, scores[i], stages[i])
    end
    return results, stats
end

function _device_walk_start(backend, vm, source, max_init_attempts, log_density)
    vm.dimension > 0 ||
        throw(ArgumentError("Random-walk samplers require latent variables"))
    score = backend._device_compile(log_density, backend._device_array(vm.sample))
    state, _ = _mh_start(max_init_attempts, "log density") do i
        x = i == 1 ? vm.sample : WrenCore.vector_parameters(vm, source)
        all(isfinite, x) || return nothing, -Inf
        device_x = backend._device_array(x)
        density = score(device_x)
        host_density = backend._to_host(density)
        host_density < Inf ||
            throw(DomainError(host_density, "Expected finite log density or -Inf"))
        return (device_x, backend._device_scalar(density), x, typeof(host_density)),
        host_density
    end
    return state
end

_device_walk_factor(algorithm, step, d) = Matrix(Diagonal(fill(step, d)))
_device_walk_factor(::Val{:rwm}, step, d) = fill(step, d)

_device_walk_proposal(x, factor::AbstractMatrix, noise) = x + factor * noise
_device_walk_proposal(x, factor::AbstractVector, noise) = x .+ factor .* noise

function _device_walk_score(log_density, x)
    density = log_density(x)
    return ifelse(all(isfinite, x), density, oftype(density, -Inf))
end

function _device_walk_move(backend, algorithm, log_density, rng, x, density, factor)
    noise = Random.randn(rng, eltype(x), length(x))
    proposal = _device_walk_proposal(x, factor, noise)
    proposed_density = _device_walk_score(log_density, proposal)
    log_alpha = min(0, proposed_density - density)
    accepted = log(Random.rand(rng)) < log_alpha
    invalid = !(proposed_density < Inf)
    return proposal, proposed_density, ifelse(accepted, 1, 0), noise, log_alpha, invalid
end

function _device_walk_warmup(backend, algorithm, log_density, state, i, warmup)
    (; rng, x, density, factor, adaptation, invalid) = state
    proposal, proposed_density, stage, noise, log_alpha, bad_density =
        _device_walk_move(backend, algorithm, log_density, rng, x, density, factor)
    x = ifelse.(stage != 0, proposal, x)
    density = ifelse(stage != 0, proposed_density, density)
    factor, adaptation = _device_adapt_walk(
        backend,
        algorithm,
        factor,
        adaptation,
        x,
        noise,
        log_alpha,
        i,
        warmup,
    )
    invalid = invalid | bad_density | !all(isfinite, factor)
    return (; rng, x, density, factor, adaptation, invalid), nothing
end

function _device_walk_sample(backend, algorithm, log_density, state)
    (; rng, x, density, factor, invalid) = state
    proposal, proposed_density, stage, _, _, bad_density =
        _device_walk_move(backend, algorithm, log_density, rng, x, density, factor)
    x = ifelse.(stage != 0, proposal, x)
    density = ifelse(stage != 0, proposed_density, density)
    return (; rng, x, density, factor, invalid=invalid | bad_density), (x, density, stage)
end

_device_walk_adaptation(backend, algorithm, x, scale, variance) = nothing

function _device_adapt_walk(
    backend,
    algorithm,
    factor,
    ::Nothing,
    x,
    noise,
    log_alpha,
    i,
    n,
)
    return factor, nothing
end
