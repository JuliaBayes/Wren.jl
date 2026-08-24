# [Rewriting models](@id rewriting)

The right-hand side of any tilde-statement in a completed model can be replaced with a different distribution, a submodel, or a fixed value.

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
Runtime values from the caller's scope enter through `$` interpolation, described below.

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

## Interpolating runtime values

An interpolation `$value` evaluates in the caller's scope at rewrite time and splices the resulting value into any expression position, whether as a whole right-hand side, embedded inside a larger expression, or as the value a site is fixed to:

```@example rewriting
fitted = Gamma(2.0, 3.0)
mu0 = 4.0

g = @rewrite base() begin
    x ~ $fitted
    y ~ Normal($mu0, 2.0)
end
```

```@example rewriting
rand(g)
```

```@example rewriting
xhat = 0.75

@rewrite chained() begin
    x = $xhat
end
```

## Rewriting inside a submodel

A dotted target names a site inside the submodel at its head, so rewriting reaches through the tree without touching the submodel's definition:

```@example rewriting
deeper = @rewrite m begin
    x.b ~ Exponential(1.0)
end
```

```@example rewriting
rand(deeper)
```

The outer site holds the rewritten inner model, whose own statements read as-written:

```@example rewriting
deeper[@vn(x)]
```

Deeper paths (`x.y.a`) and fixing (`x.b = 0.5`) work the same way.

## The `rewrite` function

The macro uses the [`rewrite`](@ref) function under the hood, and it can also be called directly to splice in a runtime value, just as `$` does in the macro:

```@example rewriting
m2 = rewrite(base(), @vn(x) => Normal(0, 2))
```

`model[vn => value]` is shorthand for this, by analogy with the mathematical substitution ``M[x := e]``, and `model[vn]` reads the current right-hand side of a site:

```@example rewriting
m2[@vn(x)]
```

Prefixed names target sites inside submodels, so the same property paths work across conditioning, `@rewrite` blocks, the function form, and the reader:

```@example rewriting
rewrite(m, @vn(x.b) => Exponential(1.0))[@vn(x.b)]
```

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

[`remove_plate`](@ref) and [`add_plate`](@ref) convert a model between its `@plate` and plain `for` loop spellings, and [`Wren.set_arg`](@ref) replaces a model argument while reusing the generated code:

```@example rewriting
lm = remove_plate(plated(3))
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

## Benchmarking

Rewriting happens at construction time, so a rewritten model evaluates at the same speed as its hand-written equivalent.
The model `m` from above rewrote `base` with `x ~ inner()`, which we can compare against the same model written directly:

```@example rewriting
using Enzyme

@wren function hand()
    x ~ inner()
    y ~ Normal()
end

@bmodel hand() AutoEnzyme(; mode=Reverse)
```

```@example rewriting
@bmodel m AutoEnzyme(; mode=Reverse)
```

The same holds for a family collapsed into a single vectorised statement:

```@example rewriting
@wren function hand_vec(N)
    mu ~ Normal()
    y ~ product_distribution(fill(Exponential(2.0), N))
end

@bmodel hand_vec(3) AutoEnzyme(; mode=Reverse)
```

```@example rewriting
pmx = @rewrite plated(3) begin
    y ~ product_distribution(fill(Exponential(2.0), N))
end

@bmodel pmx AutoEnzyme(; mode=Reverse)
```

The function form differs in one respect.
Its value is stored as a hidden model argument, so the value is built once at rewrite time rather than on every evaluation, which can shift where work happens relative to the literal statement.

And for a rewrite reaching inside a submodel:

```@example rewriting
@wren function inner_exp()
    a ~ Normal()
    b ~ Exponential(1.0)
end

@wren function hand_nested()
    x ~ inner_exp()
    y ~ Normal()
end

@bmodel hand_nested() AutoEnzyme(; mode=Reverse)
```

```@example rewriting
@bmodel deeper AutoEnzyme(; mode=Reverse)
```
