"""
    AbstractLatentSource

An abstract type representing somewhere from which the values of latent variables can be
obtained.

Wren provides several subtypes of `AbstractLatentSource`:

- [`FromPrior`](@ref): samples latent variables from their prior distributions
- [`FromParams`](@ref): provides latent variables in a `VarNamedTuple`
- [`FromVector`](@ref): provides a vector of flattened latent variables, together with
  information about how to convert them back into the original space

Custom subtypes of this can be created to customise model execution. To be a valid
`AbstractLatentSource`, a type must implement

- [`get_latent`](@ref)(source, vn, dist) to obtain a latent variable from the source, and
- [`scope`](@ref) to obtain a latent source that applies only to a subset of variables.
  This is used in submodel evaluation: if a model contains `a ~ submodel()`, then the
  latent source used to evaluate `submodel` is obtained by calling `scope(source, @vn(a))`.
"""
abstract type AbstractLatentSource end

"""
    get_latent(
        source::AbstractLatentSource,
        vn::VarName,
        dist,
        ::Val{sitename}
    ) where {sitename} -> TransformedValue

Obtain a value for the latent variable `vn` from `source`.

`dist` is the prior distribution of the variable, which occurs on the right-hand side of the
model's tilde-statement. `sitename` is a compile-time Symbol identifying the sampling site
in the model source code.

This function should return a `TransformedValue` which specifies both the value as well as
the way to convert the value back to the model's original space. Note that this allows for
lazy evaluation of the transformation, and thus `get_latent` should be as lazy as possible,
and avoid performing any transformations on the value before returning it.
"""
function get_latent end

"""
    scope(source::AbstractLatentSource, vn::VarName) -> AbstractLatentSource

Restrict `source` such that it only provides values for sub-variables of `vn`. This is used
when evaluating submodels with a prefix `vn`.

This is best explained through an example. Suppose you have a model with

```julia
@wren function inner()
    x ~ Normal(0, 1)
end
@wren function outer()
    a ~ inner()
end
```

When evaluating `outer`, the latent source may contain a value for the variable `a.x`. For
example, you might have `FromParams(VarNamedTuple(a=VarNamedTuple(x=1.0)))`. However, when
evaluating `inner`, the submodel does not know about the prefix `a`: Wren will call `scope`
to obtain a subsetted latent source which only contains the variable `x`.
"""
function scope end
