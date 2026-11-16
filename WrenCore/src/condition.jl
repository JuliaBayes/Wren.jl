"""
    conditioned(model::Model)::NamedTuple

Return the current set of values that `model` is conditioned on.
"""
conditioned(model::Model) = model.conditioned

"""
    condition(model::Model, values::NamedTuple)::Model

Condition `model` on the values in `values`, returning a new model. The original model is
not modified. Each key conditions a whole variable, including every indexed or field
tilde-statement on that variable. Partial conditioning is not supported.

For a submodel, the value under its namespace is a `NamedTuple` of the child's
conditioned variables, for example `model | (child=(y=data,),)`.
Values replace existing keys as in `merge(::NamedTuple, ::NamedTuple)`, including a
nested submodel namespace. Nested values are not merged recursively.
"""
function condition(model::Model, values::NamedTuple)
    new_conditioned = merge(conditioned(model), values)
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned=new_conditioned,
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end

"""
    Base.:|(model::Model, values::NamedTuple)::Model

`model | values` is an alias for [`condition(model, values)`](@ref).
"""
Base.:|(model::Model, values::NamedTuple) = condition(model, values)

"""
    decondition(model::Model, vars::Union{Symbol,VarName}...)::Model

Remove whole variables from this model's conditioning, returning a new model.
Use symbols for top-level names, for example `decondition(model, :y)`, and qualified
`VarName`s for child variables, for example `decondition(model, @vn(child.y))` or
`decondition(model, @vn(children[1].y))`. With no selectors, all conditioning is removed.

Removing a parent override does not remove a child model's own conditioning.
Components of an ordinary observed array or record cannot be made latent individually.

Variables observed with `@observe` are part of the model's structure rather than its
conditioning, and are unaffected.
"""
function decondition(model::Model, vars::Union{Symbol,VarName}...)
    new_conditioned = if isempty(vars)
        NamedTuple()
    else
        foldl(vars; init=conditioned(model)) do values, var
            optic =
                var isa Symbol ? VarNames.Property{var}() : VarNames.varname_to_optic(var)
            _decondition(values, optic)
        end
    end
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned=new_conditioned,
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end

_decondition(values::NamedTuple, ::VarNames.Iden) = NamedTuple()
function _decondition(values::NamedTuple, optic::VarNames.Property{sym}) where {sym}
    haskey(values, sym) || return values
    if optic.child isa VarNames.Iden
        return Base.structdiff(values, NamedTuple{(sym,)})
    end
    child = _decondition(getproperty(values, sym), optic.child)
    # Keep empty parents: deleting a structured observation's root would make it latent.
    return merge(values, NamedTuple{(sym,)}((child,)))
end
function _decondition(values, optic::Union{VarNames.Property,VarNames.Index})
    head, tail = VarNames.ohead(optic), VarNames.otail(optic)
    return Accessors.set(values, head, _decondition(head(values), tail))
end
_decondition(values, ::VarNames.Iden) =
    throw(ArgumentError("decondition requires a whole variable or submodel namespace"))
