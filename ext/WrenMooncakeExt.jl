module WrenMooncakeExt

using Mooncake: Mooncake
using Wren: Wren

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(Wren._get_range_and_transform),
    Vararg,
}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(Wren.target_transform),Vararg}

end # module WrenMooncakeExt
