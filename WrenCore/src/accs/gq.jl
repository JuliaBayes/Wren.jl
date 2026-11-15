const GQ_ACCNAME = :GeneratedQuantities

"""
    GeneratedQuantitySpec

Selects which generated quantities (`@gq` statements) are evaluated and recorded. One of
[`AllGQs`](@ref), [`SomeGQs`](@ref) or [`NoGQs`](@ref).
"""
abstract type GeneratedQuantitySpec end

"""
    AllGQs()

Select every generated quantity.
"""
struct AllGQs <: GeneratedQuantitySpec end

"""
    NoGQs()

Select no generated quantities.
"""
struct NoGQs <: GeneratedQuantitySpec end

"""
    SomeGQs(names::Tuple{Vararg{Symbol}})

Select the generated quantities named in `names`. The name of a submodel, such as `:a` in
`a ~ sub(...)`, selects every generated quantity of that submodel.
"""
struct SomeGQs{names} <: GeneratedQuantitySpec
    SomeGQs(names::Tuple{Vararg{Symbol}}) = new{names}()
end

_gq_requested(::AllGQs, ::Val) = true
_gq_requested(::NoGQs, ::Val) = false
_gq_requested(::SomeGQs{names}, ::Val{sym}) where {names,sym} = sym in names

"""
    GeneratedQuantityAccumulator(spec::GeneratedQuantitySpec=AllGQs())

Record the values of the `@gq` statements selected by `spec` in a `VarNamedTuple`.
"""
struct GeneratedQuantityAccumulator{S<:GeneratedQuantitySpec,V<:VarNamedTuple} <:
       AbstractAccumulator
    spec::S
    values::V
end
function GeneratedQuantityAccumulator(spec::GeneratedQuantitySpec=AllGQs())
    return GeneratedQuantityAccumulator(spec, VarNamedTuple())
end

function _gq_requested(acc::GeneratedQuantityAccumulator, sym::Val)
    return _gq_requested(acc.spec, sym)
end

"""
    is_accumulating_gq(accs::AccumulatorTuple, ::Val{sym})

Return `true` if `accs` contains a [`GeneratedQuantityAccumulator`](@ref) that selects `sym`.
"""
function is_accumulating_gq(accs::AccumulatorTuple, sym::Val)
    hasacc(accs, Val(GQ_ACCNAME)) || return false
    return _gq_requested(getacc(accs, Val(GQ_ACCNAME)), sym)
end

accumulator_name(::Type{<:GeneratedQuantityAccumulator}) = GQ_ACCNAME

function Base.copy(acc::GeneratedQuantityAccumulator)
    return GeneratedQuantityAccumulator(acc.spec, copy(acc.values))
end

function accumulate_assume!!(
    acc::GeneratedQuantityAccumulator,
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    @nospecialize(sitename::Val)
)
    return acc
end
function accumulate_observe!!(
    acc::GeneratedQuantityAccumulator,
    val,
    vn,
    right,
    template,
    @nospecialize(sitename::Val)
)
    return acc
end

"""
    accumulate_gq!!(acc::AbstractAccumulator, value, vn::VarName)

Update and then return `acc` when a `@gq` statement records `value` under `vn`.
"""
accumulate_gq!!(acc::AbstractAccumulator, value, vn::VarName) = acc
function accumulate_gq!!(acc::GeneratedQuantityAccumulator, value, vn::VarName)
    new_values = VarNames.templated_setindex!!(acc.values, value, vn, VarNames.NoTemplate())
    return GeneratedQuantityAccumulator(acc.spec, new_values)
end

"""
    accumulate_gq!!(at::AccumulatorTuple, value, vn::VarName)

Update all the accumulators of `at` via `accumulate_gq!!`.
"""
@generated function accumulate_gq!!(
    at::AccumulatorTuple{N,T},
    value,
    vn::VarName,
) where {N,T}
    names = fieldnames(T)
    calls = [:(accumulate_gq!!(at.nt.$(names[i]), value, vn)) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        AccumulatorTuple(NamedTuple{$names}(($(calls...),)))
    end
end

function _zero(acc::GeneratedQuantityAccumulator)
    return GeneratedQuantityAccumulator(acc.spec, empty(acc.values))
end
reset(acc::GeneratedQuantityAccumulator) = _zero(acc)
split(acc::GeneratedQuantityAccumulator) = _zero(acc)
function combine(acc1::GeneratedQuantityAccumulator, acc2::GeneratedQuantityAccumulator)
    return GeneratedQuantityAccumulator(acc1.spec, merge(acc1.values, acc2.values))
end

function scope(acc::GeneratedQuantityAccumulator, ::VarName{sym}) where {sym}
    submodel_spec = _gq_requested(acc, Val{sym}()) ? AllGQs() : NoGQs()
    return GeneratedQuantityAccumulator(submodel_spec, acc.values)
end

function combine_with_namespace(
    parent::GeneratedQuantityAccumulator,
    child::GeneratedQuantityAccumulator,
    vn::VarName,
    template,
)
    isempty(child.values) && return parent
    nested_values =
        VarNames.templated_setindex!!(VarNamedTuple(), child.values, vn, template)
    return GeneratedQuantityAccumulator(parent.spec, merge(parent.values, nested_values))
end
