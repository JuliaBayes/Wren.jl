# Wren's parameter-space metadata is passive. Keep these rules benchmark-local until Differ
# is released and Wren can provide a package extension for it.
function Differ.frule!!(
    ::Differ.Dual{typeof(Wren._get_range_and_transform)},
    vecinfo::Differ.Dual,
    vn::Differ.Dual,
)
    result = Wren._get_range_and_transform(Differ.primal(vecinfo), Differ.primal(vn))
    return Differ.Dual(result, Differ.NoTangent())
end
function Differ.frule!!(
    ::Differ.Dual{typeof(Wren.target_transform)},
    parameter_space::Differ.Dual,
    vn::Differ.Dual,
)
    result = Wren.target_transform(Differ.primal(parameter_space), Differ.primal(vn))
    return Differ.Dual(result, Differ.NoTangent())
end
function Differ.rrule!!(
    ::Differ.CoDual{typeof(Wren._get_range_and_transform),Differ.NoFData},
    ::Differ.AbstractCtx,
    vecinfo::Differ.CoDual,
    vn::Differ.CoDual,
)
    result = Wren._get_range_and_transform(Differ.primal(vecinfo), Differ.primal(vn))
    pullback(_) = (Differ.NoRData(), Differ.NoRData(), Differ.NoRData())
    return Differ.CoDual(result, Differ.NoFData()), pullback
end

function Differ.rrule!!(
    ::Differ.CoDual{typeof(Wren.target_transform),Differ.NoFData},
    ::Differ.AbstractCtx,
    parameter_space::Differ.CoDual,
    vn::Differ.CoDual,
)
    result = Wren.target_transform(Differ.primal(parameter_space), Differ.primal(vn))
    pullback(_) = (Differ.NoRData(), Differ.NoRData(), Differ.NoRData())
    return Differ.CoDual(result, Differ.NoFData()), pullback
end
