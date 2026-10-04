# [Argument copying](@id copies)

When you define a Wren model, you might see that some arguments are marked as being _copied_, such as `y` below:

```@example copying
using Wren, Wren.Dists

@wren function f(y)
    a ~ Normal()
    for i in eachindex(y)
        y[i] ~ Normal(a)
    end
end

Y = zeros(5)
model = f(Y)
```

There are two places where Wren can copy model arguments:

 1. When the model itself is **constructed**, for example, in the call to `f(Y)` above.

    Wren _always_ copies arguments at this point, to ensure that whatever the model does, it does not mutate data that were passed into it.

 2. When the model is **evaluated**, for example, when calculating log-densities, etc.
    This path is much more critical for performance as it is called many times during inference, and copying can be expensive.

    By default, Wren copies many arguments during evaluation
