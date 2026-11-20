using LinearAlgebra: diag, tril

const COMPILE_LOCK = ReentrantLock()

_device_array(x) = Reactant.to_rarray(x)
_device_scalar(x::Number) = Reactant.to_rarray(x; track_numbers=Number)
_device_rng(rng) = Reactant.ReactantRNG(_device_array(rand(rng, UInt64, 2)))
_device_wait(x) = Reactant.synchronize(x)
_device_wait(x::NamedTuple) = Reactant.synchronize(values(x))
_to_host(x::Number) = x

function _device_log_density(vm::WrenCore.VecModel)
    f = vm.lp_func
    return WrenCore._VecModelLP(
        f.evaluation_fn,
        _device_array(f.model_args),
        _device_array(f.conditioned),
        f.enzyme_copy_argnames,
        WrenCore.EvalContext{true}(),
        f.lp_function,
        f.parameter_space,
        f.vecinfo,
        f.accs,
    )
end

function _device_compile(f, args...)
    return lock(COMPILE_LOCK) do
        Reactant.@compile f(args...)
    end
end

function _device_if(yes, no, condition, args...)
    Reactant.@trace track_numbers=false if condition
        result = yes(args...)
    else
        result = no(args...)
    end
    return result
end

function _device_cholesky(covariance)
    factor = tril(Reactant.Ops.cholesky(covariance; lower=true))
    return ifelse.(all(>(0), diag(factor)), factor, eltype(factor)(NaN))
end

function _device_scan(step, state, history, n, offset=0)
    # The step may hold device arrays for the model arguments.
    loop = (; step, state)
    Reactant.@trace track_numbers=false for j in 1:n
        i = offset + j
        next, datum = loop.step(loop.state, i)
        loop = (; step=loop.step, state=next)
        _device_record!(history, datum, i)
    end
    return loop.state, history
end

_device_record!(::Nothing, ::Nothing, i) = nothing

function _device_record!(history::Tuple, datum::Tuple, i)
    foreach((h, x) -> _device_record!(h, x, i), history, datum)
    return nothing
end

function _device_record!(history::AbstractArray, datum, i)
    indices = ntuple(_ -> Colon(), ndims(history) - 1)
    Reactant.@allowscalar history[indices..., i] = datum
    return nothing
end
