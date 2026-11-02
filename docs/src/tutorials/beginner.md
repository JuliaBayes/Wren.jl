# Probabilistic programming

This page is intended to be a very gentle introduction to what probabilistic programming is all about.

If you are already familiar with other probabilistic programming languages, you may want to skip ahead to the [Syntax and semantics](@id syntax) page.

We assume here that you have a basic understanding of probability distributions (e.g. the normal and Bernoulli distributions), and that you can read Julia code.

## Probabilistic functions

'Probabilistic programming' really just means that we can represent the idea of probability or nondeterminism in our programming language.
There are many ways of doing this.
The simplest is to just write a function that calls `rand()` directly:

```@example beginner
using Wren: logdensityof
using Wren.Dists: Normal
using Random

function probabilistic_basic(rng=Random.default_rng())
    logp = 0.0
    x = rand(rng, Normal(0, 1))
    logp += logdensityof(Normal(0, 1), x)
    return (; x, logp)
end

probabilistic_basic()
```

Each time you call this function with a different random number generator, it will return a different value for `x` and `logp`.

Wren does this using the `@wren` macro, which lets you write code that is simpler, but still represents the same idea:

```@example beginner
using Wren

@wren function probabilistic()
    x ~ Normal(0, 1)
    # and so on
end

model = probabilistic()
rand_with_logp(model)
```

Technically speaking, Wren is not a full programming language in and of itself: one might call it a *domain-specific language* (DSL), which is embedded in Julia.

## Inference

Probabilistic models allow us to write down a relationship between a model's _variables_ — its parameters, which we can't directly see — and the _observations_ which we do see.
Specifying this relationship allows us to work backwards, using the observations to _infer_ the values of the parameters that could have caused them.

The best way to understand this is to construct a simple (and sadly very realistic) example.
To get to work each morning, I have to take [the train](https://en.wikipedia.org/wiki/London_Underground) to work.
Although the train is usually pretty reliable, there are often delays.
This introduces some randomness into my commute time.

For example, we might say that the time I spend on the train is 30 minutes, plus some `delay` that's between 0 and 10 minutes.
Assuming that every possible value for the delay is equally likely, we could write this as:

```math
t = 30 + d, \quad d \sim \text{Uniform}(0, 10)
```

Here, `Uniform(0, 10)` represents the uniform distribution between the bounds 0 and 10, and the `~` symbol means "is distributed as".

Unfortunately for me, I might also sometimes oversleep!
Let's say that the chance of me oversleeping is 10%.
Then `s`, which is a boolean indicating whether or not I overslept, can be represented as a Bernoulli distribution:

```math
s \sim \text{Bernoulli}(0.1)
```

If I oversleep, then I will leave the house 10 minutes late.
Altogether, that means that my total commute time `c` can be represented as:

```math
c = 30 + d + 10s
```

Let's turn this into a Wren model.
Our model has two _parameters_: `s` and `d`.
For now, don't worry too much about the exact syntax; however, you should notice that the code we wrote is very similar to the mathematical expressions above.

```@example beginner
using Wren
using Wren.Dists

@wren function time_to_office()
    s ~ Bernoulli(0.1)          # Oversleeping
    d ~ Uniform(0, 10)          # Train delay
    c = 30 + d + (s ? 10 : 0)   # Total commute time
    return c
end
```

This model already lets us generate random samples of my commute time.

TODO This needs #268

```@example beginner
```

## Further reading

For more information about probabilistic programming and Bayesian inference, you might want to check out the following resources:

TODO
