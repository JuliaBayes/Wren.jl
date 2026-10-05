# Wren benchmarks

## Benchmark suite

`suite/` benchmarks Wren's primal, Enzyme reverse, and Mooncake reverse on a set of
PosteriorDB models, and saves the results to a JSON file in `suite/results/`, named
`<UTC timestamp>-<short git sha>.json`:

```sh
julia --project=bench bench/suite/run.jl
```

CI runs this on every push to `main` and on release tags, and saves the file to `runs/` on
the `benchmark-results` branch. To benchmark a PR, add the `benchmark` label to it (to run
again later, remove the label and add it back).

`web/` is a small website that shows one run as a table, or compares two runs (pick a
'tip' and a 'base'; cells are coloured by how much faster or slower tip is, and hovering
shows the base time).

It can only be run locally right now. However, when the repo is public, we can
host this on GitHub Pages:

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
