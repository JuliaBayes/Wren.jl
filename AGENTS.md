# Instructions for Agents

  - **Always** use a temporary environment with Wren and/or WrenCore dev'd into it to run code.
    This avoids adding deps into Wren itself.
    For example, Distributions is not a dependency of Wren and should not be added as such.

  - Sometimes you will need to add new dependencies. **ALWAYS DO THIS BY CALLING Pkg.jl**; do NOT
    edit the `Project.toml` manually as you will often insert the wrong UUID!

  - `DensityInterface.logdensityof`, most relevant ADTypes, and `Enzyme.Reverse` are re-exported by WrenCore.
    No need to import them.

  - When doing MCMC sampling **always** use the `progress_backend=()` keyword argument unless you are working on the progress logging code itself.

# Overview

Wren consists of two subpackages, Wren and WrenCore.

WrenCore contains the functionality for defining and evaluating models.
Wren is a higher-level package that wraps WrenCore and combines it with inference methods and post-processing.

The top-level Project.toml defines a 'workspace', which ensures that when you load WrenCore, its dependencies are resolved in a way that is compatible with Wren.
You should not make any changes to the top-level Project.toml.

The documentation in `docs/` is also part of this workspace, and is not tied to either Wren or WrenCore; it reflects the project as a whole.

# Running models

```
using WrenCore, Distributions

@wren function f(a)
    x ~ Normal(a)
end

model = f(1.0)
```

  - `parameters(model)` draws a sample from the model.
  - `simulate(model)` draws a sample plus the components of log density.

For more information about model syntax, see `docs/src/syntax.md`.

# Vectorised models

```
vecmodel = VecModel(model, AutoEnzyme(; mode=Reverse)) # or any other adtype
v = vector_parameters(vecmodel) # draws a vector
```

  - `lp(vecmodel, v)` computes the log density.
  - `lp_and_grad(vecmodel, v)` computes a (logdensity, gradient) tuple.

# Benchmarking models

Wren provides nice macros for this:

```
@bmodel model         # no AD
@bmodel model adtype  # with AD
```
