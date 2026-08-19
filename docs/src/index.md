# Wren.jl

Wren is a library for probabilistic programming in Julia.

## Installation

As Wren is not yet registered in the Julia package registry, you can install it directly from GitHub using:

```julia
import Pkg;
Pkg.add(url="https://github.com/JuliaBayes/Wren.jl")
```

## Overview

Wren models are defined using a `@model` macro, similar to other Julia PPLs:

```@example intro
using Wren, Distributions

# Define a model
@model function f()
    x ~ Normal()
end
```

```@example intro
# Instantiate a model
model = f()
```

```@example intro
# Sample from the prior
rand(model)
```
