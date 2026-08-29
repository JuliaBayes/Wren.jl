"""
    AbstractAccumulator

An abstract type for accumulators.

An accumulator is an object that may change its value at every `tilde_assume!!` or
`tilde_observe!!` call based on the random variable in question. The obvious examples of
accumulators are the log prior and log likelihood. Other examples might be a variable that
counts the number of observations in a trace, or a list of the names of random variables
seen so far.

An accumulator type `T <: AbstractAccumulator` must implement the following methods:
- `accumulator_name(acc::T)` or `accumulator_name(::Type{T})`
- `accumulate_observe!!(acc::T, val, vn, dist, template, sitename)`
- `accumulate_assume!!(acc::T, val, tval, vn, dist, logjac, template, sitename)`
- `reset(acc::T)`
- `Base.copy(acc::T)`

In these functions:
- `val` is the new value of the random variable supplied by a latent source (always in
  the distribution's original space), or the value on the left-hand side of an observe
  statement.
- `tval` is the [`TransformedValue`](@ref) produced by interpreting the value supplied by
  the latent source in the selected parameter space.
- `dist` is the distribution on the RHS of the tilde statement.
- `logjac` is the forward log absolute determinant of the Jacobian for the transform from
  the distribution's original space to the representation selected by the parameter
  space.
- `vn` is the `VarName` that is on the left-hand side of the tilde-statement. If the
  tilde-statement is a literal observation like `0.0 ~ Normal()`, then `vn` is `nothing`.
- `template` is a value that conveys the shape of the top-level symbol in `vn`, and is
  used specifically for accumulators that carry VarNamedTuples.
- `sitename` is a compile-time `Symbol` identifying the sampling site in the model source
  code. Each distinct tilde statement has a unique `sitename`.

To be able to work with multi-threading, it should also implement:
- `split(acc::T)`
- `combine(acc::T, acc2::T)`

See the documentation for each of these functions for more details.
"""
abstract type AbstractAccumulator end

"""
    accumulator_name(acc::AbstractAccumulator)

Return a Symbol which can be used as a name for `acc`.

The name has to be unique in the sense that an `AccumulatorTuple` can only have one
accumulator for each name. The most typical case, and the default implementation, is that
the name only depends on the type of `acc`, not on its value.
"""
accumulator_name(acc::AbstractAccumulator) = accumulator_name(typeof(acc))

"""
    accumulate_observe!!(acc::AbstractAccumulator, right, left, vn, template, sitename)

Update `acc` in a `tilde_observe!!` call. Returns the updated `acc`.

See [`AbstractAccumulator`](@ref) for the meaning of the arguments.

`accumulate_observe!!` may mutate `acc`, but not any of the other arguments.

See also: [`accumulate_assume!!`](@ref)
"""
function accumulate_observe!! end

"""
    accumulate_assume!!(acc::AbstractAccumulator, val, tval, vn, dist, logjac, template, sitename)

Update `acc` in a `tilde_assume!!` call. Returns the updated `acc`.

See [`AbstractAccumulator`](@ref) for the meaning of the arguments.

`accumulate_assume!!` may mutate `acc`, but not any of the other arguments.

See also: [`accumulate_observe!!`](@ref)
"""
function accumulate_assume!! end

"""
    accumulate_addloglikelihood!!(acc::AbstractAccumulator, value, sitename)

Update `acc` when an `@addloglikelihood` statement is encountered. `value` is the scalar
log-likelihood increment. `sitename` is a compile-time `Symbol` identifying the statement.

Returns the updated `acc`. The default implementation is a no-op.

See also: [`accumulate_observe!!`](@ref), [`accumulate_assume!!`](@ref)
"""
accumulate_addloglikelihood!!(acc::AbstractAccumulator, value, sitename) = acc

"""
    reset(acc::AbstractAccumulator)

Return a new accumulator like `acc`, but with its contents reset to the state that they
should be at the beginning of model evaluation.

Note that this may in general have very similar behaviour to [`split`](@ref), and may share
the same implementation, but the difference is that `split` may in principle happen at any
stage during model evaluation, whereas `reset` is only called at the beginning of model
evaluation.
"""
function reset end

@doc """
    Base.copy(acc::AbstractAccumulator)

Create a new accumulator that is a copy of `acc`, without aliasing (i.e., this should
behave conceptually like a `deepcopy`).
""" Base.copy

"""
    split(acc::AbstractAccumulator)

Return a new accumulator like `acc` suitable for use in a forked thread.

The returned value should be such that `combine(acc, split(acc))` is equal to `acc`. This is
used in the context of multi-threading where different threads may accumulate independently
and the results are then combined.

Note that this may in general have very similar behaviour to [`reset`](@ref), but is
semantically different. See [`reset`](@ref) for more details.

See also: [`combine`](@ref)
"""
function split end

"""
    combine(acc::AbstractAccumulator, acc2::AbstractAccumulator)

Combine two accumulators which have the same type (but may, in general, have different type
parameters). Returns a new accumulator of the same type.

See also: [`split`](@ref)
"""
function combine end

"""
    scope(acc::AbstractAccumulator, vn::VarName)

Narrow `acc` for evaluation inside a submodel scoped by `vn`. The default is a no-op.
"""
scope(acc::AbstractAccumulator, ::VarName) = acc

"""
    combine_with_namespace(parent, child, vn, template)

Embed the contents of `child` beneath `vn`, then combine it with `parent`.
"""
function combine_with_namespace end

"""
    AccumulatorTuple{N,T<:NamedTuple}

A collection of accumulators, stored as a `NamedTuple` of length `N`

This is defined as a separate type to be able to dispatch on it cleanly and without method
ambiguities or conflicts with other `NamedTuple` types. We also use this type to enforce the
constraint that the name in the tuple for each accumulator `acc` must be
`accumulator_name(acc)`, and these names must be unique.

The constructor can be called with a tuple or a `VarArgs` of `AbstractAccumulators`. The
names will be generated automatically. One can also call the constructor with a `NamedTuple`
but the names in the argument will be discarded in favour of the generated ones.
"""
struct AccumulatorTuple{N,T<:NamedTuple}
    nt::T

    function AccumulatorTuple(t::T) where {N,T<:NTuple{N,AbstractAccumulator}}
        names = map(accumulator_name, t)
        nt = NamedTuple{names}(t)
        return new{N,typeof(nt)}(nt)
    end
end

@generated function hasacc(::AccumulatorTuple{N,T}, ::Val{accname}) where {N,T,accname}
    names = fieldnames(T)
    if accname in names
        return :(true)
    else
        return :(false)
    end
end

AccumulatorTuple(accs::Vararg{AbstractAccumulator}) = AccumulatorTuple(accs)
AccumulatorTuple(nt::NamedTuple) = AccumulatorTuple(tuple(nt...))
AccumulatorTuple(at::AccumulatorTuple) = at

function pretty_print(io::IO, at::AccumulatorTuple, prefix::String)
    naccs = length(at)
    if isempty(at.nt)
        print(io, "AccumulatorTuple with 0 accumulators")
        return nothing
    end
    println(
        io,
        "AccumulatorTuple with ",
        naccs,
        naccs == 1 ? " accumulator:" : " accumulators",
    )
    for (i, (name, acc)) in enumerate(pairs(at.nt))
        tree_symbol = i == naccs ? "└─ " : "├─ "
        key_name = string(name)
        print(io, prefix * tree_symbol)
        printstyled(io, key_name; color=:cyan)
        print(io, " => ")
        show(io, acc)
        if i < naccs
            println(io)
        end
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", at::AccumulatorTuple)
    pretty_print(io, at, "")
    return nothing
end
Base.getindex(at::AccumulatorTuple, idx) = at.nt[idx]
Base.length(::AccumulatorTuple{N}) where {N} = N
Base.iterate(at::AccumulatorTuple, args...) = iterate(at.nt, args...)
function Base.haskey(at::AccumulatorTuple, ::Val{accname}) where {accname}
    # @inline to ensure constant propagation can resolve this to a compile-time constant.
    @inline return haskey(at.nt, accname)
end
Base.keys(at::AccumulatorTuple) = keys(at.nt)
Base.:(==)(at1::AccumulatorTuple, at2::AccumulatorTuple) = at1.nt == at2.nt
Base.hash(at::AccumulatorTuple, h::UInt) = Base.hash((AccumulatorTuple, at.nt), h)
Base.copy(at::AccumulatorTuple) = AccumulatorTuple(map(copy, at.nt))

function Base.convert(::Type{AccumulatorTuple{N,T}}, accs::AccumulatorTuple{N}) where {N,T}
    return AccumulatorTuple(convert(T, accs.nt))
end

"""
    setacc!!(at::AccumulatorTuple, acc::AbstractAccumulator)

Add `acc` to `at`. Returns a new `AccumulatorTuple`.

If an `AbstractAccumulator` with the same `accumulator_name` already exists in `at` it is
replaced. `at` will never be mutated, but the name has the `!!` for consistency with the
corresponding function for `AbstractVarInfo`.
"""
function setacc!!(at::AccumulatorTuple, acc::AbstractAccumulator)
    accname = accumulator_name(acc)
    new_nt = merge(at.nt, NamedTuple{(accname,)}((acc,)))
    return AccumulatorTuple(new_nt)
end

"""
    getacc(at::AccumulatorTuple, ::Val{accname})

Get the accumulator with name `accname` from `at`.
"""
function getacc(at::AccumulatorTuple, ::Val{accname}) where {accname}
    return at[accname]
end

"""
    deleteacc!!(at::AccumulatorTuple, ::Val{accname})

Delete the accumulator with name `accname` from `at`. Returns a new `AccumulatorTuple`.
"""
function deleteacc!!(
    accs::AccumulatorTuple{N,<:NamedTuple{names}},
    ::Val{T},
) where {N,names,T}
    return AccumulatorTuple(NamedTuple{filter(x -> x != T, names)}(accs.nt))
end

@inline function Base.map(func::Function, at::AccumulatorTuple)
    return AccumulatorTuple(map(func, at.nt))
end

function combine_with_namespace(
    parent::AccumulatorTuple,
    child::AccumulatorTuple,
    vn::VarName,
    template,
)
    keys(parent) == keys(child) ||
        throw(ArgumentError("Cannot combine AccumulatorTuples with different accumulators"))
    combined = map(values(parent.nt), values(child.nt)) do parent_acc, child_acc
        combine_with_namespace(parent_acc, child_acc, vn, template)
    end
    return AccumulatorTuple(combined)
end

@generated function scope(
    accs::AccumulatorTuple{N,T},
    vn::VarName,
) where {N,T}
    names = fieldnames(T)
    calls = [:(scope(accs.nt.$(names[i]), vn)) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        AccumulatorTuple(NamedTuple{$names}(($(calls...),)))
    end
end

"""
    map_accumulator(func::Function, at::AccumulatorTuple, ::Val{accname})

Update the accumulator with name `accname` in `at` by calling `func` on it.

Returns a new `AccumulatorTuple`.
"""
function map_accumulator(
    func::Function,
    at::AccumulatorTuple,
    ::Val{accname},
) where {accname}
    # Would like to write this as
    # return Accessors.@set at.nt[accname] = func(at[accname], args...)
    # for readability, but that one isn't type stable due to
    # https://github.com/JuliaObjects/Accessors.jl/issues/198
    #
    # TODO(penelopeysm) Check if the above can be fixed
    new_val = func(at[accname])
    new_nt = merge(at.nt, NamedTuple{(accname,)}((new_val,)))
    return AccumulatorTuple(new_nt)
end

"""
    accumulate_assume!!(at::AccumulatorTuple, val, tval, vn, dist, logjac, template, sitename)

Update all the accumulators of `at` by calling `accumulate_assume!!` on them.
"""
@generated function accumulate_assume!!(
    at::AccumulatorTuple{N,T},
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    sitename,
) where {N,T}
    names = fieldnames(T)
    calls = [
        :(accumulate_assume!!(
            at.nt.$(names[i]),
            val,
            tval,
            vn,
            right,
            logjac,
            template,
            sitename,
        )) for i in 1:N
    ]
    return quote
        $(Expr(:meta, :inline))
        AccumulatorTuple(NamedTuple{$names}(($(calls...),)))
    end
end

"""
    accumulate_observe!!(at::AccumulatorTuple, val, vn, right, template, sitename)

Update all the accumulators of `at` by calling `accumulate_observe!!` on them.
"""
@generated function accumulate_observe!!(
    at::AccumulatorTuple{N,T},
    val,
    vn,
    right,
    template,
    sitename,
) where {N,T}
    names = fieldnames(T)
    calls = [
        :(accumulate_observe!!(at.nt.$(names[i]), val, vn, right, template, sitename))
        for i in 1:N
    ]
    return quote
        $(Expr(:meta, :inline))
        AccumulatorTuple(NamedTuple{$names}(($(calls...),)))
    end
end

"""
    accumulate_addloglikelihood!!(at::AccumulatorTuple, value, sitename)

Update all the accumulators of `at` by calling `accumulate_addloglikelihood!!` on them.
"""
@generated function accumulate_addloglikelihood!!(
    at::AccumulatorTuple{N,T},
    value,
    sitename,
) where {N,T}
    names = fieldnames(T)
    calls =
        [:(accumulate_addloglikelihood!!(at.nt.$(names[i]), value, sitename)) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        AccumulatorTuple(NamedTuple{$names}(($(calls...),)))
    end
end
