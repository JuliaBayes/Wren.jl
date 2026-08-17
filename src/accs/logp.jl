using Preferences: @load_preference, @set_preferences!

basetypeof(::T) where {T} = Base.typename(T).wrapper

"""
    NoLogProb <: Real

Singleton type that represents the absence of a log probability value. This is used as the
default type parameter for `LogProbAccumulator` when no log probability value is needed, to
avoid defining a concrete type such as `Float64` that would cause unwanted type promotion
when accumulating log probabilities of other types (e.g., `Float32`).

Adding anything to `NoLogProb()` returns the other thing. In other words, `NoLogProb` is a
true additive identity which additionally preserves types.
"""
struct NoLogProb <: Real end
Base.zero(::Type{NoLogProb}) = NoLogProb()
Base.convert(::Type{T}, ::NoLogProb) where {T<:Number} = zero(T)
Base.promote_rule(::Type{NoLogProb}, ::Type{T}) where {T<:Number} = T
Base.iszero(::NoLogProb) = true
Base.hash(::NoLogProb, h::UInt) = hash(0.0, h)
Base.:(+)(::NoLogProb, ::NoLogProb) = NoLogProb()
Base.:(-)(::NoLogProb, ::NoLogProb) = NoLogProb()

const FLOAT_TYPE_PREF_KEY = "floattype"

"""
    LogProbType

The default type used for log-probabilities in Wren.jl. This is a compile-time constant
that can be set via [`set_logprob_type!`](@ref), which under the hood uses Preferences.jl.

Note that this does not prevent computations within the model from promoting the
log-probability to a different type. In essence, `LogProbType` specifies the *lowest*
possible type that log-probabilities can be, and Wren promises to not insert any extra
operations that would cause this to be promoted to a higher type. However, Wren cannot
guard against user code inside models.

For example, in:

```julia
@model f() = x ~ Normal(0.0, 1.0)
```

the log-probability of the model will always be promoted to `Float64`, regardless of the
value of `LogProbType`, because the logpdf of `Normal(0.0, 1.0)` is a `Float64`. On the
other hand, in:

```julia
@model f() = x ~ Normal(0.0f0, 1.0f0)
```

the log-probability of the model will be `Float32` if `LogProbType` is `Float32` or lower.
"""
const LogProbType = let
    logp_pref = @load_preference(FLOAT_TYPE_PREF_KEY, "f64")
    if logp_pref == "f64"
        Float64
    elseif logp_pref == "f32"
        Float32
    elseif logp_pref == "f16"
        Float16
    elseif logp_pref == "min"
        NoLogProb
    else
        error("Unsupported log probability preference: $logp_pref")
    end
end

"""
    set_logprob_type!(::Type{T}) where {T}

Set the log probability type for Wren.jl, [`Wren.LogProbType`](@ref), to `T`.
Permitted values are `Float64`, `Float32`, `Float16`, and `NoLogProb`. The default in
Wren is `Float64`.

`NoLogProb` is a special type that is the "lowest" possible float type. This means that the
log probability will be promoted to whatever type the model dictates. This is a totally
unintrusive option, which can be useful if you do not know in advance what log probability
type you are targeting, or want to troubleshoot a model to see what type the log probability
is being promoted to. However, this can also cause type stability issues and performance
degradations, so we generally recommend setting a specific log probability type if you know
what type you want to target.

This function uses Preferences.jl to set a compile-time constant, so you will need to
restart your Julia session for the change to take effect.
"""
function set_logprob_type!(::Type{T}) where {T}
    new_pref = if T == Float64
        "f64"
    elseif T == Float32
        "f32"
    elseif T == Float16
        "f16"
    elseif T == NoLogProb
        "min"
    else
        throw(ArgumentError("Unsupported log probability type: $T"))
    end
    @set_preferences!(FLOAT_TYPE_PREF_KEY => new_pref)
    @info "Wren's log probability type has been set to $T.\nPlease note you will need to restart your Julia session for this change to take effect."
end

"""
    LogProbAccumulator{T} <: AbstractAccumulator

An abstract type for accumulators that hold a single scalar log probability value.

Every subtype of `LogProbAccumulator` must implement
* A method for `logp` that returns the scalar log probability value that defines it.
* A single-argument constructor that takes a `logp` value.
* `accumulator_name`, `accumulate_assume!!`, and `accumulate_observe!!` methods like any
    other accumulator.

`LogProbAccumulator` provides implementations for other common functions, like convenience
constructors, `copy`, `show`, `==`, `isequal`, `hash`, `split`, and `combine`.

This type has no great conceptual significance, it just reduces code duplication between
types like LogPriorAccumulator, LogJacobianAccumulator, and LogLikelihoodAccumulator.
"""
abstract type LogProbAccumulator{T<:Real} <: AbstractAccumulator end

"""
    LogProbAccumulator{T}()

Create a new `LogProbAccumulator` accumulator with the log prior initialized to `zero(T)`.

    LogProbAccumulator()

Create a new `LogProbAccumulator{Wren.LogProbType}` accumulator.
"""
(::Type{AccType})() where {T<:Real,AccType<:LogProbAccumulator{T}} = AccType(zero(T))
(::Type{AccType})() where {AccType<:LogProbAccumulator} = AccType{LogProbType}()

Base.copy(acc::LogProbAccumulator) = acc

function Base.show(io::IO, acc::LogProbAccumulator)
    return print(io, "$(string(basetypeof(acc)))($(repr(logp(acc))))")
end

# Note that == and isequal are different, and equality under the latter should imply
# equality of hashes. Both of the below implementations are also different from the default
# implementation for structs.
function Base.:(==)(acc1::LogProbAccumulator, acc2::LogProbAccumulator)
    return accumulator_name(acc1) === accumulator_name(acc2) && logp(acc1) == logp(acc2)
end

function Base.isequal(acc1::LogProbAccumulator, acc2::LogProbAccumulator)
    return basetypeof(acc1) === basetypeof(acc2) && isequal(logp(acc1), logp(acc2))
end

Base.hash(acc::T, h::UInt) where {T<:LogProbAccumulator} = hash((T, logp(acc)), h)

_zero(::Tacc) where {Tlogp,Tacc<:LogProbAccumulator{Tlogp}} = Tacc(zero(Tlogp))
reset(acc::LogProbAccumulator) = _zero(acc)
split(acc::LogProbAccumulator) = _zero(acc)
function combine(acc::LogProbAccumulator, acc2::LogProbAccumulator)
    if basetypeof(acc) !== basetypeof(acc2)
        msg = "Cannot combine accumulators of different types: $(basetypeof(acc)) and $(basetypeof(acc2))"
        throw(ArgumentError(msg))
    end
    return basetypeof(acc)(logp(acc) + logp(acc2))
end
function combine_with_namespace(
    acc::LogProbAccumulator,
    acc2::LogProbAccumulator,
    ::VarName,
    template,
)
    return combine(acc, acc2)
end

acclogp(acc::LogProbAccumulator, val) = basetypeof(acc)(logp(acc) + val)
acclogp(acc::LogProbAccumulator{NoLogProb}, val) = basetypeof(acc)(val)

function Base.convert(
    ::Type{AccType},
    acc::LogProbAccumulator,
) where {T,AccType<:LogProbAccumulator{T}}
    return AccType(convert(T, logp(acc)))
end

function promote_for_threadsafe_eval(
    acc::LogProbAccumulator,
    ::Type{Tparam_eltype},
) where {Tparam_eltype}
    return basetypeof(acc)(convert(Tparam_eltype, logp(acc)))
end

"""
    LogPriorAccumulator{T<:Real} <: LogProbAccumulator{T}

An accumulator that tracks the cumulative log prior during model execution.

Note that the log prior stored in here is always calculated based on unlinked
parameters, i.e., the value of `logp` is independent of whether tha VarInfo is
linked or not.

# Fields
$(TYPEDFIELDS)
"""
struct LogPriorAccumulator{T<:Real} <: LogProbAccumulator{T}
    "the scalar log prior value"
    logp::T
end

logp(acc::LogPriorAccumulator) = acc.logp

accumulator_name(::Type{<:LogPriorAccumulator}) = :LogPrior

function accumulate_assume!!(acc::LogPriorAccumulator, val, vn, right, template)
    return acclogp(acc, DensityInterface.logdensityof(right, val))
end
accumulate_observe!!(acc::LogPriorAccumulator, val, vn, right, template) = acc

"""
    LogJacobianAccumulator{T<:Real} <: LogProbAccumulator{T}

An accumulator that tracks the cumulative log Jacobian (technically,
log(abs(det(J)))) during model execution. Specifically, J refers to the
Jacobian of the _link transform_, i.e., from the space of the original
distribution to unconstrained space.

!!! note
    This accumulator is only incremented if the variable is transformed by a
    link function, i.e., if the VarInfo is linked (for the particular
    variable that is currently being accumulated). If the variable is not
    linked, the log Jacobian term will be 0.

    In general, for the forward Jacobian ``\\mathbf{J}`` corresponding to the
    function ``\\mathbf{y} = f(\\mathbf{x})``,

    ```math
    \\log(q(\\mathbf{y})) = \\log(p(\\mathbf{x})) - \\log (|\\mathbf{J}|)
    ```

    and correspondingly:

    ```julia
    getlogjoint_internal(vi) = getlogjoint(vi) - getlogjac(vi)
    ```

# Fields
$(TYPEDFIELDS)
"""
struct LogJacobianAccumulator{T<:Real} <: LogProbAccumulator{T}
    "the logabsdet of the link transform Jacobian"
    logjac::T
end

logp(acc::LogJacobianAccumulator) = acc.logjac

accumulator_name(::Type{<:LogJacobianAccumulator}) = :LogJacobian

function accumulate_assume!!(acc::LogJacobianAccumulator, val, vn, right, template)
    return acc
    # TODO: implement logjac
    # return acclogp(acc, logjac)
end
accumulate_observe!!(acc::LogJacobianAccumulator, val, vn, right, template) = acc

"""
    LogLikelihoodAccumulator{T<:Real} <: LogProbAccumulator{T}

An accumulator that tracks the cumulative log likelihood during model execution.

# Fields
$(TYPEDFIELDS)
"""
struct LogLikelihoodAccumulator{T<:Real} <: LogProbAccumulator{T}
    "the scalar log likelihood value"
    logp::T
end

logp(acc::LogLikelihoodAccumulator) = acc.logp

accumulator_name(::Type{<:LogLikelihoodAccumulator}) = :LogLikelihood

function accumulate_assume!!(acc::LogLikelihoodAccumulator, val, vn, right, template)
    return acc
end
function accumulate_observe!!(acc::LogLikelihoodAccumulator, val, vn, right, template)
    return acclogp(acc, DensityInterface.logdensityof(right, val))
end
