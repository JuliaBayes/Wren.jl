const GQ_ACCNAME = :GeneratedQuantities

"""
    All()

Select every generated quantity, as in `generated_quantities(model, chain; to_collect=All())`.
"""
struct All end

"""
    GQAccumulator()
    GQAccumulator(names::Tuple{Vararg{Symbol}})

An accumulator that records the values of `@gq` statements in a `VarNamedTuple`. With no
argument it requests every generated quantity; with `names` it requests only those whose
name is in `names`. A generated quantity that is not requested is not evaluated.

Generated quantities of a submodel `a ~ sub(...)` are recorded under `a` (e.g. `a.z`), and
are all requested when `:a` is.
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

Return `true` if `accs` holds a [`GQAccumulator`](@ref) that requests the generated quantity
`sym`. The result depends only on the types of the arguments, so it is a compile-time
constant.
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

Update `acc` when a `@gq` statement records `value` under `vn`. Returns the updated `acc`.
The default implementation is a no-op; [`GQAccumulator`](@ref) stores `value`.
"""
accumulate_gq!!(acc::AbstractAccumulator, value, vn::VarName) = acc
function accumulate_gq!!(acc::GQAccumulator{Names}, value, vn::VarName) where {Names}
    new_values = VarNames.templated_setindex!!(acc.values, value, vn, VarNames.NoTemplate())
    return GQAccumulator{Names}(new_values)
end

"""
    accumulate_gq!!(at::AccumulatorTuple, value, vn::VarName)

Update all the accumulators of `at` by calling `accumulate_gq!!` on them.
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
