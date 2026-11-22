# posteriordb

```
cd bench/adhoc
julia --project=. posteriordb/run.jl [--reactant]
```

This folder runs a series of 12 PosteriorDB benchmarks, comparing Wren and Turing.
With `--reactant`, Wren is also benchmarked with Reactant (this is slow, so it's off by
default).

The Wren models come from WrenCore's PosteriorDB extension (`posteriordb_model`), the
same as in the main benchmark suite. The Turing models in `models/` are written to match
them. Both Wren and Turing are timed with the fastest sample, and BLAS is limited to one
thread.

The Turing models declare their variables in the same order as the Wren models, so that
both can be timed at the same parameter vector (timings can depend on the point). The script
checks that the two log densities agree there, and if they don't, it warns and times Turing
at a point drawn from its prior instead.

## The models

| Posterior                              | dim | Family                               |
|:-------------------------------------- | ---:|:------------------------------------ |
| `Rate_1_data-Rate_1_model`             | 1   | Binomial proportion                  |
| `GLM_Binomial_data-GLM_Binomial_model` | 3   | Binomial regression                  |
| `gp_pois_regr-gp_regr`                 | 3   | Gaussian process regression          |
| `GLM_Poisson_Data-GLM_Poisson_model`   | 4   | Poisson regression                   |
| `kidiq-kidscore_interaction`           | 5   | Linear regression with interaction   |
| `sblri-blr`                            | 6   | Bayesian linear regression           |
| `wells_data-wells_dae_inter_model`     | 7   | Logistic regression with interaction |
| `arK-arK`                              | 7   | Autoregressive series                |
| `eight_schools-eight_schools_centered` | 10  | Centred hierarchical normal          |
| `GLMM_Poisson_data-GLMM_Poisson_model` | 45  | Poisson mixed model                  |
| `rats_data-rats_model`                 | 65  | Hierarchical linear growth           |
| `Mh_data-Mh_model`                     | 388 | Capture-recapture with heterogeneity |
