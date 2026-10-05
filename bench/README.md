# Wren benchmarks

## Benchmark suite

`suite/` benchmarks Wren on a set of PosteriorDB models.

By default, this is intended to be used for 'production' benchmarking. In other
words, the benchmarks used here are saved in the repo's history (specifically
the `benchmark-results` branch) so that we can keep a record of performance
over time.

To run the benchmarks locally (e.g. to try out local changes), run:

```sh
julia --project=bench bench/suite/run.jl --no-save
```

### Instructions for production benchmarking

The git sha, commit date, Julia version, package versions and machine details
are filled in automatically.

By default, runs are labelled with just the short SHA. However, for version
releases we'll want to label the run with the version number, so that we can
easily see the performance of each release. To do so, run:

```sh
julia --project=bench bench/suite/run.jl --label "WrenCore@X.Y.Z"
```

Production benchmarks aren't currently run on CI because timings are too
inconsistent. (If one day we have dedicated machines, we could do that
instead.) Instead, the production benchmarks are currently manually run on
Penny's laptop.

### The website

`web/` is a small website that shows one run as a table, or compares two runs
(pick a 'tip' and a 'base'; cells are coloured by how much faster or slower tip
is, and hovering shows the base time).

Currently, because the repo is private, there are some limitations. In particular:

  - It's not hosted online, you have to view it locally.
  - You have to fetch the JSON benchmark results yourself.

When the repo is public, we can host this on GitHub Pages, and make the website
fetch results over HTTP from GitHub.

Anyway, to view this locally:

```sh
cd bench/web
pnpm install
scripts/fetch-results.sh   # copy runs from the benchmark-results branch
pnpm dev
```

You can also load JSON files from disk with the 'Load file…' buttons.

## Ad hoc benchmarks

`adhoc/` has other one-off benchmarks (e.g. Wren vs Turing), with their own environment.
See `adhoc/README.md`.
