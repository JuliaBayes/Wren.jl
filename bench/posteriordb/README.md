# TL;DR

```
cd bench
julia --project=. posteriordb/run.jl
```

# PosteriorDB benchmarks

`posteriordb.jl` compares [Wren](https://github.com/JuliaBayes/Wren.jl) with the
matching Stan implementation on ten PosteriorDB posteriors, and places the
recorded Turing results from `benchmarks/posteriordb.md` beside them.

The environment develops Wren and ProbabilityMeasures from `~/.julia/dev`, so it
benchmarks whatever is checked out there.

Run from the repository root:

```sh
julia --project=benchmark_wren -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark_wren benchmark_wren/posteriordb.jl
```

Run one model, or check the Reactant backend against Enzyme:

```sh
julia --project=benchmark_wren benchmark_wren/posteriordb.jl sblri-blr
julia --project=benchmark_wren benchmark_wren/posteriordb.jl --verify sblri-blr
```

`--logdensity-only` omits gradients, and `--stan-only` or `--wren-only` times one
implementation. `PDB_BENCH_MARKDOWN` chooses the report path, `PDB_BENCH_OUTPUT`
saves the raw timings as tab-separated data, and `PDB_BENCH_SHARDS` with the
one-based `PDB_BENCH_SHARD` splits the run across processes; merge the
checkpoints with `--merge`. BridgeStan uses `BRIDGESTAN_PATH` when it is set.

## The models

Ten posteriors, chosen to span parameter dimension from 1 to 388 and to cover
distinct model families: conjugate binomial, binomial and Poisson GLMs, Gaussian
process regression, linear regression, an autoregressive series, a centred
hierarchical normal, a Poisson mixed model, and a capture-recapture model with a
marginalised latent indicator.

| Posterior                              | dim | Family                               |
|:-------------------------------------- | ---:|:------------------------------------ |
| `Rate_1_data-Rate_1_model`             | 1   | Binomial proportion                  |
| `GLM_Binomial_data-GLM_Binomial_model` | 3   | Binomial regression                  |
| `gp_pois_regr-gp_regr`                 | 3   | Gaussian process regression          |
| `GLM_Poisson_Data-GLM_Poisson_model`   | 4   | Poisson regression                   |
| `kidiq-kidscore_interaction`           | 5   | Linear regression with interaction   |
| `sblri-blr`                            | 6   | Bayesian linear regression           |
| `arK-arK`                              | 7   | Autoregressive series                |
| `eight_schools-eight_schools_centered` | 10  | Centred hierarchical normal          |
| `GLMM_Poisson_data-GLMM_Poisson_model` | 45  | Poisson mixed model                  |
| `Mh_data-Mh_model`                     | 388 | Capture-recapture with heterogeneity |

## What the numbers do and do not mean

**Timings are comparable; log densities are not.** Two things separate this
benchmark from `benchmarks/posteriordb.jl`, which times DynamicPPL and Stan at a
matched parameter realization and verifies the densities and gradients against
each other.

First, Stan and Wren are each timed at their own reproducible draw. Wren's
unconstrained coordinates differ from Stan's, and this benchmark does not map
between them.

Second, ProbabilityMeasures has no truncation, no improper priors and no beta
measure, so several priors are substituted:

| Stan prior                         | Wren prior       |
|:---------------------------------- |:---------------- |
| `truncated(Cauchy(0, s); lower=0)` | `Exponential(s)` |
| `truncated(Normal(0, s); lower=0)` | `Exponential(s)` |
| `Flat()`                           | `Normal(0, 100)` |

A substituted model has a different posterior from the Stan program it is named
after. Each substitution is marked at the tilde statement in
`posteriordb_models.jl`. Two further rewrites there are exact and are marked as
such: `Beta(1, 1)` is `Uniform(0, 1)`, and a binomial likelihood on the logit
scale is `ProductBinomialLogit`.

Four of the ten models — `Rate_1_model`, `GLM_Binomial_model`,
`GLM_Poisson_model` and `Mh_model` — use no substituted prior and therefore keep
the Stan posterior.

What is verified is that Wren's Reactant backend reproduces its Enzyme backend's
log density and gradient, at random points of Wren's own unconstrained space.
`--verify` runs that check. It exits non-zero over the whole catalog, because
`gp_pois_regr-gp_regr` has no Reactant gradient to compare against; the other
nine agree to around 1e-13.

## Reading the table

`dim` is the parameter count, checked against Stan's unconstrained dimension so a
translation with the wrong number of parameters fails rather than being timed.

The eval columns are one log-density evaluation: `Wren` in plain Julia and
`WrenRct` compiled through Reactant. The gradient columns are one
value-and-gradient evaluation: `EzyRvs` is Enzyme in reverse mode and `RctRvs` is
Reactant driving Enzyme in reverse mode. `RctRvs / Stan` divides the Reactant
gradient time by Stan's; lower is better. A backend that fails on a model reports
`err` for that cell and does not stop the run.

`gp_pois_regr-gp_regr` is one such row: its covariance is factorised, and
Reactant's Enzyme pipeline has no adjoint for `stablehlo.cholesky`, so both of
its Reactant cells report `err` and only Enzyme measures its gradient.

The Reactant columns carry a fixed per-call cost of roughly 25–75 µs that does
not shrink with the model: `Rate_1_model` at one parameter and `Mh_model` at 388
sit in the same band. Every posterior here evaluates in microseconds or less
under plain Julia, so that cost dominates all ten rows and the Reactant ratios
say more about dispatch overhead than about the compiled kernels. Reading them as
a verdict on Reactant would need models large enough to amortise it — these are
not.

Table 2 sets the ratios measured here beside the Turing and Mooncake ratios
recorded in `benchmarks/posteriordb.md`. That report was produced on different
hardware, so only the ratios against Stan carry across — and Turing's models use
the Stan priors, while several of Wren's are substituted.

Rows whose times fall below about 100 ns are dominated by timer floor and fixed
overhead; for those, the raw times say more than the ratios.

## Layout

  - `posteriordb.jl` — the runner: model selection, timing, checkpoints, report.
  - `posteriordb_models.jl` — the ten Wren models and their PosteriorDB data
    adapters.
  - `pmproducts.jl` — products of independent ProbabilityMeasures observations.
    ProbabilityMeasures has no product measure, and Wren's `@batch` lowers to a
    measure that reads its arguments one element at a time, which Reactant
    rejects. These measures evaluate as array arithmetic and a reduction instead.
