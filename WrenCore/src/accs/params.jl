const PARAMS_ACCNAME = :Params

# TODO(mhauru) The deepcopy here is quite unfortunate. It is needed so that the model body
# can go mutating the object without that in turn mutating the value stored in the
# accumulator, which should be as it was at `~` time. Could there be a way around this?
_safe_copy(val) = deepcopy(val)
# collect is much faster than deepcopy on views, and for our purposes is the same (returns a
# copy of data that is not aliased to the original).
# See https://github.com/TuringLang/DynamicPPL.jl/pull/1350
_safe_copy(val::SubArray) = collect(val)

struct GetParams end
Base.copy(g::GetParams) = g
(g::GetParams)(val, tval, vn, right, logjac) = _safe_copy(val)

"""
    ParamsAccumulator() <: AbstractAccumulator

An accumulator that keeps tracks of the model parameters exactly as they are seen in the
model.
"""
function ParamsAccumulator()
    return VNTAccumulator{PARAMS_ACCNAME}(GetParams())
end

#################################################################

# Debug version of ParamsAccumulator: it does the same thing as ParamsAccumulator, but additionally
# errors if a value is set twice. This is used in check_model.
struct DebugGetParams
    repeated_vns::Set{VarName}
end
Base.copy(d::DebugGetParams) = DebugGetParams(copy(d.repeated_vns))
function DebugParamsAccumulator()
    return VNTAccumulator{PARAMS_ACCNAME}(DebugGetParams(Set{VarName}()))
end

# Unfortunately we have to overload accumulate_assume!! since we need to use the
# templated_setindex_no_overwrite!! function
function accumulate_assume!!(
    acc::VNTAccumulator{PARAMS_ACCNAME,<:Any,DebugGetParams},
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    @nospecialize(sitename::Val),
)
    new_val = _safe_copy(val)
    # The exception catching is probably slow, but it's ok since it only happens inside
    # check_model.
    new_vnt = try
        VarNames.templated_setindex_no_overwrite!!(acc.values, new_val, vn, template)
    catch e
        # Don't error immediately, save it for later.
        if e isa VarNames.MustNotOverwriteError
            push!(acc.f.repeated_vns, e.target_vn)
            VarNames.templated_setindex!!(acc.values, new_val, vn, template)
        else
            rethrow(e)
        end
    end
    return update_values(acc, new_vnt)
end

function combine(
    acc1::VNTAccumulator{PARAMS_ACCNAME,<:Any,DebugGetParams},
    acc2::VNTAccumulator{PARAMS_ACCNAME,<:Any,DebugGetParams},
)
    if acc1.f !== acc2.f
        throw(ArgumentError("Cannot combine accumulators with different functions"))
    end

    new_values = acc1.values
    for (vn, val) in pairs(acc2.values)
        top_sym = VarNames.getsym(vn)
        template_from_acc2_values = get(acc2.values.data, top_sym, VarNames.NoTemplate())
        new_values = try
            VarNames.templated_setindex_no_overwrite!!(
                new_values,
                val,
                vn,
                template_from_acc2_values,
            )
        catch e
            if e isa VarNames.MustNotOverwriteError
                push!(acc1.f.repeated_vns, e.target_vn)

                # Note: if `acc1` and `acc2` have different templates
                # `templated_setindex!!` uses the structure inside `acc1`'s values.
                VarNames.templated_setindex!!(
                    new_values,
                    val,
                    vn,
                    template_from_acc2_values,
                )
            else
                rethrow(e)
            end
        end
    end
    return update_values(acc1, new_values)
end
