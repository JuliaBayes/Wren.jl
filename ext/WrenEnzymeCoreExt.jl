module WrenEnzymeCoreExt

using EnzymeCore
using Wren: Wren

@inline EnzymeCore.EnzymeRules.inactive(::typeof(Wren._get_range_and_transform), args...) =
    nothing
@inline EnzymeCore.EnzymeRules.inactive(::typeof(Wren.target_transform), args...) = nothing

end # module WrenEnzymeCoreExt
