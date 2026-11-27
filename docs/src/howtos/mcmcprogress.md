# [Displaying MCMC progress](@id mcmc-progress)

Despite our best efforts to make Wren as fast as possible, sampling will still take long enough for you to experience the progress bars from [MCMCProgress.jl](https://juliabayes.org/MCMCProgress.jl/stable/).

The goal of this short guide is to walk you through the different ways you can configure how sampling progress is displayed.

## Default progress display

By default, sampling from this model with `nuts` will use MCMCProgress' `TermBackend`.
Built on [Term.jl](https://fedeclaudi.github.io/Term.jl/stable/), it is the most visually featureful of the available backends, but also the backend that's least tested across different platforms.
You can see `TermBackend` in action at the bottom of the MCMCProgress docs' [HMC example page](https://juliabayes.org/MCMCProgress.jl/stable/hmc/).

MCMCProgress.jl offers two more backend options: `ProgressLoggingBackend` and `PlainTextBackend`.
In short, the former is a more robust alternative to `TermBackend`, while the latter is a minimal, text-based progress display.

If you want to change the different default backend for all samplers in the current Julia session, pass a backend to `MCMCProgress.set_backend!`.

## Disabling progress display

To sample without progress displays, you can either pass `progress=false` or an empty tuple to a sampler.

## Multiple progress displays

It's possible to pass multiple backends in a tuple `progress=(TermBackend(), PlainTextBackend())` to sample with multiple progress displays.
