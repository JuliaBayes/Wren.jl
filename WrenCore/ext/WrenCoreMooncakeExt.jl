module WrenCoreMooncakeExt

using Mooncake: Mooncake
using WrenCore: WrenCore

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(WrenCore._get_range_and_transform),
    Vararg,
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(WrenCore.target_transform),
    Vararg,
}

end # module WrenCoreMooncakeExt
