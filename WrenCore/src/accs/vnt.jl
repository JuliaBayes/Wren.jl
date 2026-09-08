"""
    VNTAccumulator{AccName}(f::F, values::VarNamedTuple=VarNamedTuple()) where {AccName,F}
    VNTAccumulator{AccName,KeepEmptySubmodels}(f::F, values::VarNamedTuple=VarNamedTuple())

A generic accumulator that applies a function `f` to values seen during model execution
and stores the results in a `VarNamedTuple`.

`AccName` is the name of the accumulator, and is exposed to allow users to define and use
multiple forms of `VNTAccumulator` within the same set of accumulators. In theory, each
`VNTAccumulator` with the same function `f` should use the same accumulator name. This is
not enforced.

`KeepEmptySubmodels` controls what [`combine_with_namespace`](@ref) does with a submodel
that accumulated nothing (i.e., a key `@vn(a) => VarNamedTuple()` in the parent
accumulator). When `false` (the default), the empty VNT is dropped from the parent, meaning
that `@vn(a)` will not even be a key in the parent. When `true` the empty VNT is retained.
In the absence of submodels, this option has no effect.

The function `f` should have the signature:

    f(val, tval, vn, dist, logjac) -> value_to_store

where the arguments have their usual meanings in [`accumulate_assume!!`](@ref). If a value
does not need to be accumulated, this can be signalled by returning `DoNotAccumulate()`
from `f`.

If `f` is a struct (and not a function), you also need to define `Base.copy` on `f`.
"""
struct VNTAccumulator{AccName,KeepEmptySubmodels,F,VNT<:VarNamedTuple} <:
       AbstractAccumulator
    f::F
    values::VNT
end
function VNTAccumulator{AccName,KeepEmptySubmodels}(
    f::F,
    values::VarNamedTuple=VarNamedTuple(),
) where {AccName,KeepEmptySubmodels,F}
    return VNTAccumulator{AccName,KeepEmptySubmodels,F,typeof(values)}(f, values)
end
function VNTAccumulator{AccName}(
    f::F,
    values::VarNamedTuple=VarNamedTuple(),
) where {AccName,F}
    return VNTAccumulator{AccName,false}(f, values)
end

"""
    DoNotAccumulate()

Sentinel value indicating that no accumulation should be performed for a given variable.
"""
struct DoNotAccumulate end

function Base.copy(acc::VNTAccumulator{AccName,KE}) where {AccName,KE}
    return VNTAccumulator{AccName,KE}(copy(acc.f), copy(acc.values))
end
function Base.copy(acc::VNTAccumulator{AccName,KE,F}) where {AccName,KE,F<:Function}
    return VNTAccumulator{AccName,KE}(acc.f, copy(acc.values))
end
accumulator_name(::VNTAccumulator{AccName}) where {AccName} = AccName

function update_values(
    acc::VNTAccumulator{AccName,KE},
    new_values::VarNamedTuple,
) where {AccName,KE}
    return VNTAccumulator{AccName,KE}(acc.f, new_values)
end

function accumulate_assume!!(
    acc::VNTAccumulator,
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    @nospecialize(sitename::Val),
)
    new_val = acc.f(val, tval, vn, right, logjac)
    return if new_val isa DoNotAccumulate
        acc
    else
        new_values = VarNames.templated_setindex!!(acc.values, new_val, vn, template)
        update_values(acc, new_values)
    end
end
accumulate_observe!!(
    acc::VNTAccumulator,
    val,
    vn,
    right,
    template,
    @nospecialize(sitename::Val)
) = acc

_zero(acc::VNTAccumulator) = update_values(acc, empty(acc.values))
reset(acc::VNTAccumulator) = _zero(acc)
split(acc::VNTAccumulator) = _zero(acc)
function combine(
    acc1::VNTAccumulator{AccName},
    acc2::VNTAccumulator{AccName},
) where {AccName}
    if acc1.f != acc2.f
        throw(ArgumentError("Cannot combine VNTAccumulator with different functions"))
    end
    return update_values(acc1, merge(acc1.values, acc2.values))
end
function combine_with_namespace(
    acc1::VNTAccumulator{AccName,KE},
    acc2::VNTAccumulator{AccName,KE},
    vn::VarName,
    template,
) where {AccName,KE}
    if !KE && isempty(acc2.values)
        return acc1
    end
    nested_values =
        VarNames.templated_setindex!!(VarNamedTuple(), acc2.values, vn, template)
    return combine(acc1, update_values(acc2, nested_values))
end
