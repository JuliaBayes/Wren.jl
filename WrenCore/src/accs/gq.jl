const GQ_ACCNAME = :GeneratedQuantities

"""
    All()

Select every generated quantity, as in `generated_quantities(model, chain; to_collect=All())`.
"""
struct All end

"""
    GQAccumulator()
    GQAccumulator(names::Tuple{Vararg{Symbol}})

Record the values of `@gq` statements in a `VarNamedTuple`. If `names` is not provided
it returns every generated quantity, and otherwise it returns only those in `names`.
"""
struct GQAccumulator{Names,V<:VarNamedTuple} <: AbstractAccumulator
    values::V
end
GQAccumulator{Names}(values) where {Names} = GQAccumulator{Names,typeof(values)}(values)
GQAccumulator() = GQAccumulator{:all}(VarNamedTuple())
GQAccumulator(names::Tuple{Vararg{Symbol}}) = GQAccumulator{names}(VarNamedTuple())

_gq_requested(names, sym::Symbol) = names === :all || sym in names

"""
    requests_gq(accs::AccumulatorTuple, ::Val{sym})

Return `true` if `accs` contains a [`GQAccumulator`](@ref) for `sym`.
"""
@generated function requests_gq(::AccumulatorTuple{N,T}, ::Val{sym}) where {N,T,sym}
    GQ_ACCNAME in fieldnames(T) || return :(false)
    names = fieldtype(T, GQ_ACCNAME).parameters[1]
    return _gq_requested(names, sym) ? :(true) : :(false)
end

accumulator_name(::Type{<:GQAccumulator}) = GQ_ACCNAME

Base.copy(acc::GQAccumulator{Names}) where {Names} = GQAccumulator{Names}(copy(acc.values))

function accumulate_assume!!(
    acc::GQAccumulator,
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
    acc::GQAccumulator,
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
function accumulate_gq!!(acc::GQAccumulator{Names}, value, vn::VarName) where {Names}
    new_values = VarNames.templated_setindex!!(acc.values, value, vn, VarNames.NoTemplate())
    return GQAccumulator{Names}(new_values)
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

_zero(acc::GQAccumulator{Names}) where {Names} = GQAccumulator{Names}(empty(acc.values))
reset(acc::GQAccumulator) = _zero(acc)
split(acc::GQAccumulator) = _zero(acc)
function combine(acc1::GQAccumulator{Names}, acc2::GQAccumulator) where {Names}
    return GQAccumulator{Names}(merge(acc1.values, acc2.values))
end

function scope(acc::GQAccumulator{Names}, ::VarName{sym}) where {Names,sym}
    return if _gq_requested(Names, sym)
        GQAccumulator{:all}(acc.values)
    else
        GQAccumulator{()}(acc.values)
    end
end

function combine_with_namespace(
    parent::GQAccumulator{Names},
    child::GQAccumulator,
    vn::VarName,
    template,
) where {Names}
    isempty(child.values) && return parent
    nested_values =
        VarNames.templated_setindex!!(VarNamedTuple(), child.values, vn, template)
    return GQAccumulator{Names}(merge(parent.values, nested_values))
end
