# [Rewriting models](@id rewriting)

The right-hand side of any tilde-statement in a completed model can be replaced with a different distribution, a submodel, or a fixed value.
Because Wren models store their own AST, the result is statement-for-statement identical to a model that had been written that way from the start, and the evaluation function is regenerated automatically (and cached, so repeated rewrites do not recompile).

## The `@rewrite` macro

`@rewrite` takes a model and a block of statements written like a model body, and replaces the right-hand sides of the named sites:

```@example rewriting
using Wren, Distributions

@wren function inner()
    a ~ Normal()
    b ~ Normal(a)
end

@wren function base()
    x ~ Normal()
    y ~ Normal()
end

m = @rewrite base() begin
    x ~ inner()
end
```

The displayed AST is exactly what runs.
The site `x` samples the submodel, precisely as if `base` had been declared with `x ~ inner()` all along.

```@example rewriting
rand(m)
```

Each statement must name an existing site of the model.
The replacement expressions become part of the model body, so they are re-evaluated on every model evaluation, and model arguments and module globals resolve as usual, but local variables at the `@rewrite` call site do not.
To splice in a runtime value, call [`rewrite`](@ref) directly, described below.

## Fixing a site to a value

An assignment statement in the block makes the site deterministic, in the style of the [do-operator](https://en.wikipedia.org/wiki/Do-calculus):

```@example rewriting
@wren function chained()
    x ~ Normal()
    y ~ Normal(x)
end

fixed = @rewrite chained() begin
    x = 1.5
end
```

```@example rewriting
rand(fixed)
```

The fixed site contributes no log density, so the log-joint is the density of `y ~ Normal(1.5)` alone:

```@example rewriting
(logjoint(fixed, VarNamedTuple(; y=2.0)), logdensityof(Normal(1.5), 2.0))
```

In reverse, a tilde statement makes a fixed site random again, whether the assignment came from an earlier rewrite or was written in the model body.

```@example rewriting
unfixed = @rewrite fixed begin
    x ~ Exponential(1.0)
end
```

```@example rewriting
rand(unfixed)
```

## The `rewrite` function

The macro uses the [`rewrite`](@ref) function under the hood, and it can also be called directly to splice in a runtime value rather than an expression:

```@example rewriting
m2 = rewrite(base(), @vn(x) => Normal(0, 2))
```

`model[vn => value]` is shorthand for this, by analogy with the mathematical substitution ``M[x := e]``, and `model[vn]` reads the current right-hand side of a site:

```@example rewriting
m2[@vn(x)]
```

## Rewriting inside a submodel

Rewriting targets the model's own sites, so a prefixed name such as `@vn(x.a)` does not name a site of the outer model.
To rewrite a site inside a submodel, compose the two ingredients above, rewriting the inner model value and putting it back:

```@example rewriting
nested = rewrite(m, @vn(x) => rewrite(m[@vn(x)], @vn(a) => Exponential(1.0)))
```

```@example rewriting
rand(nested)
```

Prefixed targets are a natural extension of this composition.

## Plated and looped sites

A tilde statement inside a `@plate` block or a plain `for` loop defines a site family, addressed by its name.
Targeting the family replaces its whole construct with the single given statement, exactly as if the model had been written that way; other statements in the construct's body stay in the loop, and the family's dead array allocation is removed:

```@example rewriting
@wren function plated(N)
    y = zeros(N)
    mu ~ Normal()
    @plate for i in 1:N
        y[i] ~ Normal(mu)
    end
end

pm = rewrite(plated(3), @vn(y) => product_distribution(fill(Exponential(2.0), 3)))
```

```@example rewriting
rand(pm)
```

Fixing a family to a value works the same way, in either form:

```@example rewriting
pf = @rewrite plated(3) begin
    y = [0.5, 1.0, 1.5]
end
```

```@example rewriting
rand(pf)
```

In the opposite direction, a single statement can be replaced by an elementwise construct, with the block pairing the family's array allocation with the loop.
The round trip returns to a model equivalent to the original:

```@example rewriting
elem = @rewrite pm begin
    y = zeros(N)
    @plate for i in 1:N
        y[i] ~ Exponential(2.0)
    end
end
```

```@example rewriting
rand(elem)
```

A right-hand side that references the loop index keeps its elementwise meaning, and replaces the right-hand side inside the construct:

```@example rewriting
@rewrite plated(3) begin
    y ~ Normal(mu, i)
end
```

A single element of a family (e.g. `@vn(y[1])`) is not an addressable site, since the family's elements share one statement.
Rewrite the whole family instead ([issue #19](https://github.com/JuliaBayes/Wren.jl/issues/19) tracks element-granular rewriting).

## Converting between plates and loops

[`Wren.remove_plate`](@ref) and [`Wren.add_plate`](@ref) convert a model between its `@plate` and plain `for` loop spellings, and [`Wren.set_arg`](@ref) replaces a model argument while reusing the generated code:

```@example rewriting
lm = Wren.remove_plate(plated(3))
```

```@example rewriting
rand(Wren.set_arg(lm, :N, 5))
```

## Indexed sites

A literal indexed statement such as `x[1] ~ inner` is a site in its own right, targeted by the matching `VarName`:

```@example rewriting
@wren function indexed(sub)
    x = zeros(2)
    x[1] ~ sub
    x[2] ~ Normal()
end

ind = rewrite(indexed(inner()), @vn(x[1]) => Normal())
```

```@example rewriting
rand(ind)
```

## Conditioning into a submodel

Rewriting also underlies conditioning values into a submodel from the outer model, where the conditioned submodel is rewritten in for the original.
See the [submodels page](@ref submodels) for that interface.

## Benchmarking

Rewriting happens at construction time, so a rewritten model evaluates at the same speed as its hand-written equivalent.
The model `m` from above rewrote `base` with `x ~ inner()`, which we can compare against the same model written directly:

```@example rewriting
@wren function hand()
    x ~ inner()
    y ~ Normal()
end

@bmodel hand()
```

```@example rewriting
@bmodel m
```

The same holds for gradients:

```@example rewriting
using Enzyme
@bmodel hand() AutoEnzyme(; mode=Reverse)
```

```@example rewriting
@bmodel m AutoEnzyme(; mode=Reverse)
```
