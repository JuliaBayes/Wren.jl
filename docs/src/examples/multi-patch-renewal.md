# [Multi-patch renewal modelling](@id example-multi-patch-renewal)

The time-varying reproduction number ``\mathcal{R}_t`` is the average number of new infections caused by each person infected at time ``t``, and estimating it from surveillance data is a routine part of outbreak response.
The renewal process model is the standard tool for this ([Fraser 2007](https://doi.org/10.1371/journal.pone.0000758), [Cori et al. 2013](https://doi.org/10.1093/aje/kwt133), [Gostic et al. 2020](https://doi.org/10.1371/journal.pcbi.1008409)), and it underpinned real-time transmission estimates throughout the COVID-19 pandemic ([Abbott et al. 2020](https://doi.org/10.12688/wellcomeopenres.16006.2)).
This example builds a renewal model for daily case counts with ``\mathcal{R}_t`` as the target estimand, then extends it to six spatial patches coupled by a gravity mixing matrix.
Each component is a small [submodel](@ref submodels) or plain function, written as maths and then code.
We simulate data from the prior, fit with NUTS, check parameter recovery, and forecast from a truncated fit, reusing the single-patch components in the multi-patch model.
For model development and validation practice for models of this kind, see [A Workflow for Infectious Disease Modelling](https://samabbott.co.uk/a-workflow-for-infectious-disease-modelling/paper.pdf).

```@example renewal
using Wren, Wren.Samplers
using Distributions
using CensoredDistributions
using CairoMakie, PairPlots
using LinearAlgebra: I, cholesky, Symmetric, normalize, norm
using Random, Statistics
using Enzyme
import AbstractMCMC
```

## Delay distributions

Delays link every process in infectious disease surveillance.
The generation time separates an infection from the infections it causes, the incubation period separates infection from symptom onset, and reporting delays separate events from their appearance in the data.
Any model connecting transmission to observations is therefore built from delay distributions, and misspecifying them biases both the size and the timing of the resulting estimates ([Gostic et al. 2020](https://doi.org/10.1371/journal.pcbi.1008409), [Charniga et al. 2024](https://doi.org/10.1371/journal.pcbi.1012520)).

Delays are continuous but observed at daily resolution, so they are doubly interval censored: the first event occurs at an unknown time within its day, and the second is only observed to the day.
Ignoring this censoring biases delay estimates ([Park et al. 2024](https://doi.org/10.1093/aje/kwae350)).
We use [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl) to represent the censored process.
For a delay distribution ``f``, the mass on day ``d`` is

```math
p_d = \Pr(d \le P + T < d + 1 \mid P + T \le D), \qquad
P \sim \mathrm{Uniform}(0, 1), \quad T \sim f,
```

where ``P`` is the time of the first event within its day, ``T`` is the delay to the second event, and the observable delay ``P + T`` is truncated at ``D`` days.
`double_interval_censored` composes the censoring steps, and its `pdf` at integer ``d`` gives the mass on ``[d, d + 1)``, so the daily masses already sum to one under truncation:

```@example renewal
function censored_pmf(dist; D, drop_zero=false)
    censored = double_interval_censored(dist; upper=Float64(D), interval=1.0)
    pmf = Vector{typeof(pdf(censored, 0.0))}(undef, D)
    for t in 1:D
        pmf[t] = pdf(censored, Float64(t - 1))
    end
    if drop_zero
        pmf = pmf[2:end] ./ sum(pmf[2:end])
    end
    return pmf
end
```

We take the generation time distribution from the literature, ``f_g = \mathrm{Gamma}(6.5, 0.62)`` truncated at ``D = 14`` days, giving the daily mass function ``g_s``.
Its day-zero mass is dropped, since a same-day term would put ``I_t`` on both sides of the renewal equation, and the remainder renormalised:

```@example renewal
gen_pmf = censored_pmf(Gamma(6.5, 0.62); D=14, drop_zero=true);
```

### Fitting the reporting delay

Reporting delays are estimated from linelists, and near the analysis date the longest delays have not yet been reported, so each event carries its own right truncation ([Park et al. 2024](https://doi.org/10.1093/aje/kwae350)).
We simulate three weeks of onsets with growing incidence, draw each event's observed delay from the same censored process truncated at its own window, and aggregate the linelist to unique delay and truncation pairs:

```@example renewal
true_delay = LogNormal(1.5, 0.5)
analysis_day = 21
linelist_rng = Xoshiro(2)
p_days =
    rand(linelist_rng, Distributions.Categorical(normalize(exp.(0.1 .* (0:20)), 1)), 250) .-
    1
observed(p) =
    double_interval_censored(true_delay; upper=Float64(analysis_day - p), interval=1.0)
linelist = [(rand(linelist_rng, observed(p)), Float64(analysis_day - p)) for p in p_days]
counts = Dict{Tuple{Float64,Float64},Float64}()
for row in linelist
    counts[row] = get(counts, row, 0.0) + 1.0
end
strata = collect(keys(counts))
delay_obs = first.(strata)
windows = last.(strata)
wts = [counts[k] for k in strata]
sum(wts .* delay_obs) / sum(wts)
```

The naive mean of the kept delays understates the true mean, `mean(true_delay)` ≈ 5.1 days, because long delays are missing by construction.
Fitting the censored process with each stratum's own truncation corrects this,

```math
\mu \sim \mathrm{Normal}(1, 0.5) \text{ on } [0, 3], \qquad
\sigma \sim \mathrm{Normal}(0.5, 0.25) \text{ on } [0.1, 2], \qquad
t_j \sim \mathrm{DoubleIntervalCensored}\!\left(\mathrm{LogNormal}(\mu, \sigma); D_j\right),
```

where ``t_j`` is the observed delay of stratum ``j``, ``D_j`` runs from its primary event day to the analysis day, and each stratum's likelihood is weighted by its count with CensoredDistributions' `weight`.
The bounded priors keep every warmup proposal in a numerically safe region:

```@example renewal
@wren function delay_model(mu_prior, sigma_prior, windows, weights, delays, D)
    mu ~ mu_prior
    sigma ~ sigma_prior
    for j in eachindex(delays)
        d = double_interval_censored(LogNormal(mu, sigma); upper=windows[j], interval=1.0)
        delays[j] ~ weight(d, weights[j])
    end
    return censored_pmf(LogNormal(mu, sigma); D=D)
end

delay_instance = delay_model(
    truncated(Normal(1, 0.5); lower=0, upper=3),
    truncated(Normal(0.5, 0.25); lower=0.1, upper=2),
    windows,
    wts,
    zeros(length(strata)),
    15,
)
```

The model returns the daily mass function implied by its parameters.
A model's return value at a trace comes from `evaluate` with the `FromParams` latent source (see [Working with models](@ref models)); we wrap that once and use it to check every component before it enters the composition.
Prior draws of the mass function span same-day reporting through to multi-week delays:

```@example renewal
function return_values(model, params)
    retval, _ = evaluate(
        model,
        FromParams(params),
        OriginalSpace(),
        Wren.AccumulatorTuple(Wren.ParamsAccumulator()),
    )
    return retval
end

let
    rng = Xoshiro(3)
    fig = Figure(; size=(700, 300))
    ax =
        Axis(fig[1, 1]; xlabel="days", ylabel="probability", title="Prior reporting delays")
    for _ in 1:100
        pmf = return_values(delay_instance, rand(rng, delay_instance))
        lines!(ax, 0:(length(pmf)-1), pmf; color=(:grey, 0.2))
    end
    fig
end
```

Conditioning on the observed delays turns the same instance into the fitted model:

```@example renewal
delay_fitted = delay_instance | VarNamedTuple(; delays=delay_obs)
delay_chn = nuts(Xoshiro(1), delay_fitted, 500, 2; progress=false)
delay_stats = summarystats(delay_chn)
```

A conditioned instance could slot straight into a larger composition.
Here we instead fix the reporting delay ``p_d`` at its posterior means, taken from the summary above, and return to the alternatives at the end:

```@example renewal
mu_hat = delay_stats[@vn(mu), stat=:mean]
sigma_hat = delay_stats[@vn(sigma), stat=:mean]
delay_pmf = censored_pmf(LogNormal(mu_hat, sigma_hat); D=15);
```

```@example renewal
let
    fig = Figure(; size=(700, 300))
    ax1 = Axis(fig[1, 1]; xlabel="days", ylabel="probability", title="Generation time")
    barplot!(ax1, 1:length(gen_pmf), gen_pmf; color=:steelblue)
    ax2 = Axis(fig[1, 2]; xlabel="days", title="Fitted reporting delay")
    barplot!(ax2, 0:(length(delay_pmf)-1), delay_pmf; color=:steelblue)
    fig
end
```

## A single-patch model

### Process model

The log reproduction number follows a random walk with an estimated step size ``\sigma``,

```math
Z_1 \sim \mathrm{Normal}(\mu_0, \sigma_0), \qquad
Z_t = Z_{t-1} + \sigma \varepsilon_t, \quad \varepsilon_t \sim \mathrm{Normal}(0, 1), \qquad
\mathcal{R}_t = \exp(Z_t).
```

As a submodel this is a function of the length of the walk and the two priors:

```@example renewal
@wren function random_walk(n, init_prior, step_prior)
    init ~ init_prior
    sigma ~ step_prior
    eps ~ MvNormal(zeros(n - 1), I)
    return vcat(init, init .+ sigma .* cumsum(eps))
end
```

The return value is the whole walk, which is what a parent model consumes.
We use

```math
\mu_0 = \log 1.3, \qquad \sigma_0 = 0.1, \qquad \sigma \sim \mathrm{Normal}^{+}(0, 0.05),
```

where ``\mathrm{Normal}^{+}`` is a normal truncated to be positive.
Like any Wren model the submodel can be sampled from directly:

```@example renewal
T = 56
rt_priors = (Normal(log(1.3), 0.1), truncated(Normal(0, 0.05); lower=0))
rt_model = random_walk(T, rt_priors...)
rand(Xoshiro(1), rt_model)
```

It can also be conditioned on its own parameters and sampled again, which is useful for checking a component in isolation:

```@example renewal
rand(Xoshiro(1), rt_model | VarNamedTuple(; sigma=0.1))
```

Prior draws of the implied reproduction number start around 1.3 and drift slowly, without collapsing or exploding over the eight weeks:

```@example renewal
let
    rng = Xoshiro(2)
    fig = Figure(; size=(700, 300))
    ax = Axis(fig[1, 1]; xlabel="day", ylabel="reproduction number")
    for _ in 1:100
        lines!(ax, exp.(return_values(rt_model, rand(rng, rt_model))); color=(:grey, 0.15))
    end
    hlines!(ax, [1.0]; color=:firebrick, linestyle=:dash)
    fig
end
```

Daily infection incidence ``I_t`` follows the renewal process model ([Fraser 2007](https://doi.org/10.1371/journal.pone.0000758), [Cori et al. 2013](https://doi.org/10.1093/aje/kwt133)),

```math
I_t = \mathcal{R}_t \sum_{s \ge 1} I_{t-s} g_s,
```

with infections before the first time point back-extrapolated from ``I_0`` at the exponential growth rate ``r`` implied by ``\mathcal{R}_1``,

```math
I_u = I_0 e^{r u} \ (u \le 0), \qquad
\mathcal{R}_1 \sum_{s \ge 1} g_s e^{-r s} = 1, \qquad
\log I_0 \sim \mathrm{Normal}(\log 50, 0.5).
```

The second relation links growth rates and reproduction numbers through the generation time distribution ([Wallinga and Lipsitch 2007](https://doi.org/10.1098/rspb.2006.3754)); `R_to_r` solves it with a short Newton iteration.
We use explicit loops rather than generator sums, and mark them `@inbounds`, which keeps the code fast under Enzyme's reverse-mode AD; the seeded-history branch is what guarantees every index is in range.
The clamp on ``r`` and the cap on infections keep the log-density finite at the extreme parameter values explored during warmup, where the exponential growth would otherwise overflow:

```@example renewal
function R_to_r(R, gen_pmf; newton_steps=2)
    mean_gen = zero(eltype(gen_pmf))
    for (s, g) in enumerate(gen_pmf)
        mean_gen += s * g
    end
    r = clamp((R - 1) / (R * mean_gen), -2.0, 2.0)
    for _ in 1:newton_steps
        f = zero(r)
        df = zero(r)
        for (s, g) in enumerate(gen_pmf)
            e = exp(-r * s)
            f += g * e
            df -= s * g * e
        end
        r = clamp(r - (R * f - 1) / (R * df), -2.0, 2.0)
    end
    return r
end

"Sum of `pmf`-weighted lags of `x` at `t`, reaching into the seeded history."
function lagged_sum(x, t, pmf, x0, r, first_lag)
    acc = zero(eltype(x))
    @inbounds for (i, w) in enumerate(pmf)
        s = first_lag + i - 1
        past = t - s >= 1 ? x[t-s] : x0 * exp(r * (t - s))
        acc += w * past
    end
    return acc
end

function renew(I0, R::AbstractVector, gen_pmf, r)
    I_t = similar(R)
    @inbounds for t in eachindex(R)
        I_t[t] = min(R[t] * lagged_sum(I_t, t, gen_pmf, I0, r, 1), 1e15)
    end
    return I_t
end
```

### Observation model

The observation model relates infection incidence to reported cases through the reporting delay and observation noise.
Expected reported cases convolve infections with the reporting delay distribution, and observed cases are negative binomial with overdispersion controlled by a cluster factor ``\alpha = c^2``, as in [EpiNow2](https://epiforecasts.io/EpiNow2/) ([Abbott et al. 2020](https://doi.org/10.12688/wellcomeopenres.16006.2)),

```math
Y_t = \sum_{d=0}^{D - 1} p_d I_{t-d}, \qquad
y_t \sim \mathrm{NegBin}\!\left(\mathrm{mean} = Y_t,\ \mathrm{var} = Y_t + \alpha Y_t^2\right), \qquad
c \sim \mathrm{Normal}^{+}(0, 0.1),
```

``y_t`` is `cases` in the code and ``c`` is `cluster`.
For ``t - d \le 0`` the convolution reaches into the same seeded history as the renewal recursion, so cases start at the level the mid-outbreak seeding implies.

```@example renewal
function delay_convolve(x::AbstractVector, pmf, x0, r)
    y = similar(x)
    @inbounds for t in eachindex(x)
        y[t] = lagged_sum(x, t, pmf, x0, r, 0)
    end
    return y
end

function NegativeBinomialMeanClust(mu, alpha)
    var = mu + alpha * mu^2
    p = mu / var
    r = 1 / alpha
    return NegativeBinomial(r, p; check_args=false)
end

growth_rates(R::AbstractVector, gen_pmf) = R_to_r(R[1], gen_pmf)

function expected_renewal_given_Rt(R::AbstractVector, I0, gen_pmf, delay_pmf)
    r0 = growth_rates(R, gen_pmf)
    I_t = renew(I0, R, gen_pmf, r0)
    return (; I_t, Y_t=delay_convolve(I_t, delay_pmf, I0, r0))
end
```

`check_args=false` serves the same purpose as the clamp and cap above, giving extreme proposals a log-density of ``-\infty`` rather than an error.

### The full model

The full model composes the process and observation components.
The reproduction number process enters as a submodel argument, so it can be swapped without touching the rest of the model.
The `cases` argument only declares the shape of the `cases` array (see [Syntax and semantics](@ref syntax)), and data enters later by conditioning:

```@example renewal
@wren function renewal_cases(rt_model, gen_pmf, delay_pmf, init_prior, cluster_prior, cases)
    Z_t = (log_Rt ~ rt_model)
    log_I0 ~ init_prior
    R = exp.(Z_t)
    (; I_t, Y_t) = expected_renewal_given_Rt(R, exp(log_I0), gen_pmf, delay_pmf)
    cluster ~ cluster_prior
    for t in eachindex(cases)
        cases[t] ~ NegativeBinomialMeanClust(Y_t[t], cluster^2)
    end
    return (; I_t, Y_t, R)
end

single_model(rt, n) = renewal_cases(
    rt,
    gen_pmf,
    delay_pmf,
    Normal(log(50), 0.5),
    truncated(Normal(0, 0.1); lower=0),
    zeros(Int, n),
)
model = single_model(rt_model, T)
```

`single_model` fixes the delays and priors once, so the forecasting section can rebuild the same composition at another horizon.

### Prior predictive checks

Prior predictive checks simulate data from the prior to check that model behaviour is epidemiologically plausible before fitting.
Because `cases` is latent until we condition on it, sampling from the prior simulates complete data sets:

```@example renewal
prior_chn = prior(Xoshiro(1), model, 500; progress=false)

let
    fig = Figure(; size=(700, 350))
    ax = Axis(fig[1, 1]; xlabel="day", ylabel="reported cases", yscale=log10)
    for draw in vec(prior_chn[@vn(cases)])[1:100]
        lines!(ax, max.(draw, 0.5); color=(:grey, 0.15))
    end
    fig
end
```

Simulated epidemics mostly grow slowly from around 50 reported cases a day, with a heavy upper tail from the walks that hold ``\mathcal{R}_t`` above one for the whole window, already visible in the reproduction number check above.
The component checks are what make a composed check like this one readable; when a draw looks wrong, the component it came from has its own plot to compare against.

### Fitting and parameter recovery

Fitting simulated data and checking that the known parameters are recovered validates the model and the inference together.
We take one prior draw as the truth and condition the model on its simulated cases with `|`:

```@example renewal
truth = rand(Xoshiro(132), model)
cases_data = truth[@vn(cases)]

fitted = model | VarNamedTuple(; cases=cases_data)
chn = nuts(Xoshiro(1), fitted, 1000, 2; progress=false)
chn
```

`summarystats` checks the computation before we interpret anything.
Effective sample sizes and ``\hat{R}`` near one show the chains mixed:

```@example renewal
pp_keys = [@vn(log_Rt.init), @vn(log_Rt.sigma), @vn(log_I0), @vn(cluster)]
summarystats(chn[pp_keys])
```

`pairplot` compares the posterior against the prior for the four scalar parameters, with the true values overlaid:

```@example renewal
truth_values(params, keys) = NamedTuple(Symbol(string(k)) => params[k] for k in keys)

compare_pairplot(chn, prior_chn, keys, truth) = pairplot(
    PairPlots.Series(chn[keys]; label="posterior", color=(:steelblue, 0.6)),
    PairPlots.Series(prior_chn[keys]; label="prior", color=(:grey, 0.4)),
    PairPlots.Truth(truth_values(truth, keys); label="true values"),
)

compare_pairplot(chn, prior_chn, pp_keys, truth)
```

To check recovery of the latent process we compare reproduction number trajectories.
`return_values` gives the trajectory at the truth and at each posterior draw, with `to_samples` converting the chain back to traces:

```@example renewal
posterior_returns(model, chn; thin=10) = [
    return_values(model, vnt) for
    vnt in vec(AbstractMCMC.to_samples(VarNamedTuple, chn)[1:thin:end, :])
]

R_post = stack(r.R for r in posterior_returns(fitted, chn))
R_true = return_values(model, truth).R

function ci_bands!(ax, draws)
    for (w, a) in zip((0.9, 0.6, 0.3), (0.2, 0.35, 0.5))
        lo = [quantile(view(draws, t, :), 0.5 - w / 2) for t in axes(draws, 1)]
        hi = [quantile(view(draws, t, :), 0.5 + w / 2) for t in axes(draws, 1)]
        band!(
            ax,
            axes(draws, 1),
            lo,
            hi;
            color=(:steelblue, a),
            label="$(round(Int, 100w))% CrI",
        )
    end
end

let
    fig = Figure(; size=(700, 350))
    ax = Axis(fig[1, 1]; xlabel="day", ylabel="reproduction number")
    ci_bands!(ax, R_post)
    lines!(ax, R_true; color=:firebrick, linewidth=2, label="truth")
    axislegend(ax)
    fig
end
```

Posteriors are summarised as nested 30%, 60%, and 90% credible interval bands.

### Forecasting

Fitting to a truncated series and simulating forward turns the same model into a forecast.
We refit to the first 42 days, then re-run a 56-day model at each posterior draw.
`FromParams` supplies every parameter the fit knows, and its `FromPrior` fallback draws what is missing, here the innovations beyond the fitted window and all of the cases.
The bands below are therefore posterior predictive over the fitted window and forecasts beyond it.
The walk draws its innovations as a single multivariate site that a shorter fit cannot partially fill, so the forecast uses a variant that samples them element by element:

```@example renewal
@wren function random_walk_forecast(n, init_prior, step_prior)
    init ~ init_prior
    sigma ~ step_prior
    eps = zeros(n - 1)
    for i in eachindex(eps)
        eps[i] ~ Normal()
    end
    return vcat(init, init .+ sigma .* cumsum(eps))
end

T_obs = 42
fit42 =
    single_model(random_walk(T_obs, rt_priors...), T_obs) |
    VarNamedTuple(; cases=cases_data[1:T_obs])
chn42 = nuts(Xoshiro(1), fit42, 500, 2; progress=false)

forecast_model = single_model(random_walk_forecast(T, rt_priors...), T)

predicted(rng, model, chn, key; thin=5) = stack(
    parameters(Draw(FromParams(vnt, FromPrior(rng)), model; include_log_probs=false))[key]
    for vnt in vec(AbstractMCMC.to_samples(VarNamedTuple, chn)[1:thin:end, :])
)

forecast = predicted(Xoshiro(1), forecast_model, chn42, @vn(cases))
summarystats(chn42[pp_keys])
```

`predicted` re-runs a model at each posterior trace, drawing anything the fit has not seen from its prior.

The last fortnight of the data, held out from the fit, sits inside the forecast intervals:

```@example renewal
let
    fig = Figure(; size=(700, 350))
    ax = Axis(fig[1, 1]; xlabel="day", ylabel="reported cases")
    ci_bands!(ax, forecast)
    scatter!(
        ax,
        1:T_obs,
        cases_data[1:T_obs];
        color=:black,
        markersize=5,
        label="fitted data",
    )
    scatter!(
        ax,
        (T_obs+1):T,
        cases_data[(T_obs+1):T];
        color=:firebrick,
        markersize=6,
        label="held out",
    )
    vlines!(ax, [T_obs + 0.5]; color=:grey, linestyle=:dash)
    axislegend(ax; position=:lt)
    fig
end
```

## Six coupled patches

Surveillance data are often spatially stratified.
We extend the process model to six patches whose infections mix, reusing the components above.

### Gravity mixing

Each patch has a population and a location.
The force of infection that patch ``g`` receives mixes over source patches,

```math
\Lambda_{g,t} = \sum_h K_{gh} \sum_s g_s I_{h,t-s}, \qquad
I_{g,t} = \mathcal{R}_{g,t} \Lambda_{g,t},
```

with a gravity mixing matrix ([Xia et al. 2004](https://doi.org/10.1086/422341))

```math
K_{gh} \propto \frac{\tilde p_g \tilde p_h}{d_{gh}^{\gamma}} \ (g \ne h), \qquad
K_{gg} \propto 1, \qquad \sum_h K_{gh} = 1,
```

where ``\tilde p`` is population in units of the mean and ``d_{gh}`` is the distance between patches.
Rows are normalised to sum to one so that ``K`` only says where each patch's force of infection comes from; its overall scale belongs to ``\mathcal{R}_t``, which would otherwise be confounded with it.
The distance exponent is estimated with prior ``\gamma \sim \mathrm{Normal}^{+}(1.5, 0.5)``:

```@example renewal
function gravity(pop, dist; gamma, within=1.0)
    n = length(pop)
    mean_pop = sum(pop) / n
    K = Matrix{promote_type(typeof(gamma), eltype(pop))}(undef, n, n)
    for h in 1:n, g in 1:n
        K[g, h] = if g == h
            within
        else
            (pop[g] / mean_pop) * (pop[h] / mean_pop) / dist[g, h]^gamma
        end
    end
    return K ./ sum(K; dims=2)
end

@wren function gravity_mixing(pop, dist, gamma_prior)
    gamma ~ gamma_prior
    return gravity(pop, dist; gamma)
end

rng = Xoshiro(42)
n_patches = 6
coords = [100 * Point2(rand(rng), rand(rng)) for _ in 1:n_patches]
pop = rand(rng, LogNormal(log(100_000), 0.75), n_patches)
dist = [norm(p - q) for p in coords, q in coords]
marker_size = 8 .+ 8 .* sqrt.(pop ./ mean(pop));
```

The six patches, with marker area scaled by population:

```@example renewal
let
    fig = Figure(; size=(450, 320))
    ax = Axis(fig[1, 1]; xlabel="x (km)", ylabel="y (km)", title="Patch geography")
    scatter!(ax, coords; markersize=marker_size, color=:steelblue)
    text!(ax, coords .+ Ref(Point2(2.5, 0)); text=string.(1:n_patches), fontsize=10)
    fig
end
```

Prior draws of the off-diagonal mixing weights span near-uniform coupling to sharply local transmission, with the spread at each distance coming from the population products:

```@example renewal
mix_model = gravity_mixing(pop, dist, truncated(Normal(1.5, 0.5); lower=0))

let
    rng = Xoshiro(4)
    fig = Figure(; size=(700, 300))
    ax = Axis(
        fig[1, 1];
        xlabel="distance (km)",
        ylabel="mixing weight",
        yscale=log10,
        title="Prior mixing weights",
    )
    for _ in 1:50
        K = return_values(mix_model, rand(rng, mix_model))
        for g in 1:n_patches, h in 1:n_patches
            g == h && continue
            scatter!(ax, dist[g, h], K[g, h]; color=(:grey, 0.15), markersize=3)
        end
    end
    fig
end
```

### Spatially and temporally correlated deviations

Patch reproduction numbers share the random walk from the single-patch model and add a patch deviation that evolves over weeks.
The deviation follows a damped random walk across weeks whose innovations are correlated across space,

```math
\log \mathcal{R}_{g,t} = Z_t + \delta_{g,w(t)}, \qquad
\boldsymbol{\delta}_1 = \sigma_\delta L \boldsymbol{\eta}_1, \qquad
\boldsymbol{\delta}_w = \rho \boldsymbol{\delta}_{w-1} + \sigma_\delta \sqrt{1 - \rho^2}\, L \boldsymbol{\eta}_w,
```

```math
\boldsymbol{\eta}_w \sim \mathrm{MvNormal}(\mathbf{0}, I), \qquad
L L^\top = C, \qquad C_{gh} = \exp(-d_{gh} / \ell),
```

where ``w(t)`` is the week of day ``t``.
Nearby patches drift together, individual patches diverge, and the damping ``\rho`` pulls local departures back towards the shared trend.
We fix the correlation length ``\ell = 30`` and estimate ``\sigma_\delta \sim \mathrm{Normal}^{+}(0, 0.2)`` and ``\rho \sim \mathrm{Normal}(0.8, 0.1)`` truncated to ``[0, 1]``:

```@example renewal
@wren function spatial_deviations(L_corr, n_weeks, sd_prior, damp_prior)
    sigma ~ sd_prior
    rho ~ damp_prior
    n = size(L_corr, 1)
    eta ~ MvNormal(zeros(n * n_weeks), I)
    E = reshape(eta, n, n_weeks)
    delta = similar(E)
    prev = sigma .* (L_corr * @view E[:, 1])
    delta[:, 1] = prev
    scale = sigma * sqrt(1 - rho^2)
    for w in 2:n_weeks
        prev = rho .* prev .+ scale .* (L_corr * @view E[:, w])
        delta[:, w] = prev
    end
    return delta
end

function expand_weeks(delta_w::AbstractMatrix, T)
    P, W = size(delta_w)
    out = similar(delta_w, T, P)
    @inbounds for g in 1:P, t in 1:T
        out[t, g] = delta_w[g, fld1(t, 7)]
    end
    return out
end

ell = 30.0
L_corr = cholesky(Symmetric(exp.(-dist ./ ell))).L

let
    fig = Figure(; size=(420, 320))
    ax = Axis(fig[1, 1]; xlabel="patch", ylabel="patch", title="Deviation correlation")
    hm = heatmap!(ax, exp.(-dist ./ ell); colorrange=(0, 1))
    Colorbar(fig[1, 2], hm)
    fig
end
```

Prior draws of the deviations wander to patch departures of a few tenths on the log scale and damp back towards the shared trend:

```@example renewal
n_weeks = cld(T, 7)
dev_model = spatial_deviations(
    L_corr,
    n_weeks,
    truncated(Normal(0, 0.2); lower=0),
    truncated(Normal(0.8, 0.1); lower=0, upper=1),
)

let
    rng = Xoshiro(5)
    fig = Figure(; size=(700, 300))
    ax = Axis(fig[1, 1]; xlabel="week", ylabel="deviation", title="Prior patch deviations")
    for _ in 1:20
        delta = return_values(dev_model, rand(rng, dev_model))
        for g in 1:n_patches
            lines!(ax, delta[g, :]; color=(:grey, 0.12))
        end
    end
    fig
end
```

Submodels nest, so the patch-level reproduction number process is itself a Wren model.
It reuses `random_walk` from the single-patch model for the shared trend, adds the deviations, and can be swapped or conditioned like any other submodel:

```@example renewal
@wren function patch_log_rt(trend_model, dev_model)
    Z_t = (trend ~ trend_model)
    delta = (deviations ~ dev_model)
    return (; log_R=Z_t .+ expand_weeks(delta, length(Z_t)), delta)
end
```

### Coupled process and observation models

The renewal recursion gains a method for a matrix of reproduction numbers (time by patch) and the mixing matrix, with per-patch growth rates for the seeding, and the convolution gains a matching matrix method:

```@example renewal
function growth_rates(R::AbstractMatrix, gen_pmf)
    growth = similar(R, size(R, 2))
    for h in axes(R, 2)
        growth[h] = R_to_r(R[1, h], gen_pmf)
    end
    return growth
end

function renew(I0::AbstractVector, R::AbstractMatrix, gen_pmf, K::AbstractMatrix, growth)
    T, P = size(R)
    I_t = similar(R)
    lambda = similar(I0)
    @inbounds for t in 1:T
        for h in 1:P
            lambda[h] = lagged_sum(view(I_t, :, h), t, gen_pmf, I0[h], growth[h], 1)
        end
        for g in 1:P
            pressure = zero(eltype(R))
            for h in 1:P
                pressure += K[g, h] * lambda[h]
            end
            I_t[t, g] = min(R[t, g] * pressure, 1e15)
        end
    end
    return I_t
end

function delay_convolve(x::AbstractMatrix, pmf, x0, growth)
    y = similar(x)
    @inbounds for g in axes(x, 2), t in axes(x, 1)
        y[t, g] = lagged_sum(view(x, :, g), t, pmf, x0[g], growth[g], 0)
    end
    return y
end

function expected_renewal_given_Rt(R::AbstractMatrix, I0, gen_pmf, delay_pmf, K)
    growth = growth_rates(R, gen_pmf)
    I_t = renew(I0, R, gen_pmf, K, growth)
    return (; I_t, Y_t=delay_convolve(I_t, delay_pmf, I0, growth))
end
```

The full model swaps the single-patch reproduction number process for the nested patch-level one; the delay and observation components are reused directly.
We seed the outbreak in the least populous patch, ``\log I_{0,g} \sim \mathrm{Normal}(\log 50, 0.5)`` there and ``\mathrm{Normal}(\log 0.05, 0.5)`` elsewhere, so the epidemic has to reach the other patches through the mixing matrix:

```@example renewal
@wren function patch_renewal_cases(
    rt_model,
    mix_model,
    gen_pmf,
    delay_pmf,
    seed_prior,
    cluster_prior,
    cases,
)
    rt = (log_Rt ~ rt_model)
    K_mix = (mixing ~ mix_model)
    log_I0 ~ seed_prior
    R = exp.(rt.log_R)
    (; I_t, Y_t) = expected_renewal_given_Rt(R, exp.(log_I0), gen_pmf, delay_pmf, K_mix)
    cluster ~ cluster_prior
    for g in axes(cases, 2), t in axes(cases, 1)
        cases[t, g] ~ NegativeBinomialMeanClust(Y_t[t, g], cluster^2)
    end
    return (; I_t, Y_t, R, delta=rt.delta, K_mix)
end

origin = argmin(pop)
patch_model = patch_renewal_cases(
    patch_log_rt(rt_model, dev_model),
    mix_model,
    gen_pmf,
    delay_pmf,
    MvNormal([g == origin ? log(50) : log(0.05) for g in 1:n_patches], 0.25 * I),
    truncated(Normal(0, 0.1); lower=0),
    zeros(Int, T, n_patches),
)
```

### Prior predictive checks

The composed model gets the same check as the single-patch one.
The seeded patch starts around 50 reported cases a day, the rest start near zero, and growth elsewhere can only arrive through the mixing matrix:

```@example renewal
function patch_grid(plot!; kwargs...)
    fig = Figure(; size=(700, 450))
    for g in 1:n_patches
        ax = Axis(
            fig[fld1(g, 2), mod1(g, 2)];
            title="patch $g",
            titlesize=10,
            yticks=LinearTicks(3),
            kwargs...,
        )
        plot!(ax, g)
    end
    return fig
end

patch_prior_chn = prior(Xoshiro(1), patch_model, 500; progress=false)

patch_grid(; yscale=log10) do ax, g
    for draw in vec(patch_prior_chn[@vn(cases)])[1:50]
        lines!(ax, max.(draw[:, g], 0.5); color=(:grey, 0.15))
    end
end
```

### Simulated data

One prior draw now simulates an epidemic across all six patches.
The outbreak grows in the seeded patch first, reaches the next patches around two weeks in, and takes another week to arrive in the last:

```@example renewal
patch_truth = rand(Xoshiro(8), patch_model)
patch_cases = patch_truth[@vn(cases)]

patch_grid() do ax, g
    lines!(ax, patch_cases[:, g]; color=:steelblue)
end
```

The same epidemic over the patch map, with marker area scaled by cases in the trailing week, shows the spread over space:

```@example renewal
let
    days = (14, 28, 42, 56)
    week_cases = [sum(patch_cases[(d-6):d, g]) for g in 1:n_patches, d in days]
    cmax = maximum(week_cases)
    fig = Figure(; size=(700, 200))
    for (i, d) in enumerate(days)
        ax = Axis(fig[1, i]; title="day $d", titlesize=10)
        hidedecorations!(ax)
        scatter!(
            ax,
            coords;
            markersize=4 .+ 24 .* sqrt.(week_cases[:, i] ./ cmax),
            color=:steelblue,
        )
    end
    fig
end
```

### Fitting the coupled model

The coupled posterior has harder geometry than the single-patch one, so we raise the target acceptance rate to 0.9.
Keyword arguments to `nuts` pass through to [`DynamicHMC.mcmc_with_warmup`](@extref):

```@example renewal
DHMC = Wren.Samplers.DynamicHMC
patch_fitted = patch_model | VarNamedTuple(; cases=patch_cases)
patch_chn = nuts(
    Xoshiro(1),
    patch_fitted,
    500,
    2;
    progress=false,
    warmup_stages=DHMC.default_warmup_stages(;
        stepsize_adaptation=DHMC.DualAveraging(δ=0.9),
    ),
)
patch_chn
```

The same computational checks apply:

```@example renewal
patch_keys = [
    @vn(log_Rt.trend.sigma),
    @vn(log_Rt.deviations.sigma),
    @vn(log_Rt.deviations.rho),
    @vn(mixing.gamma),
    @vn(cluster),
]
summarystats(patch_chn[patch_keys])
```

The pairplot again compares prior and posterior, now including the spatial parameters:

```@example renewal
compare_pairplot(patch_chn, patch_prior_chn, patch_keys, patch_truth)
```

The random walk step size, deviation scale, and cluster factor update strongly.
The distance exponent ``\gamma`` also updates, because the arrival time of the outbreak in each patch carries information about mixing.
With every patch seeded at once there would be no arrivals to learn from and ``\gamma`` would be weakly identified; finding out what the data can and cannot inform is one purpose of these checks.

The mixing matrix and deviations are return values of the model, so the same helpers recover their posteriors.
Because distance enters as ``d^{-\gamma}``, most transmission is within patches, so the between-patch flows below are scaled relative to the largest one.
Flows concentrate between large, nearby patches, and the deviations, averaged over weeks, vary smoothly in space, as the distance-decaying prior intends:

```@example renewal
patch_rets = posterior_returns(patch_fitted, patch_chn)
patch_true_rets = return_values(patch_model, patch_truth)

let
    delta_hat = vec(mean(stack(r.delta for r in patch_rets); dims=(2, 3)))
    K_hat = dropdims(mean(stack(r.K_mix for r in patch_rets); dims=3); dims=3)
    delta_true = vec(mean(patch_true_rets.delta; dims=2))
    K_true = patch_true_rets.K_mix
    lim = max(maximum(abs, delta_hat), maximum(abs, delta_true))
    offdiag_max(K) = maximum(K[g, h] for g in 1:n_patches, h in 1:n_patches if g != h)
    w_max = max(offdiag_max(K_hat), offdiag_max(K_true))
    function maps_row!(fig, row, K, delta, label)
        ax1 = Axis(fig[row, 1]; xlabel="x (km)", ylabel="y (km)", title="$label mixing")
        for g in 1:n_patches, h in 1:n_patches
            w = K[g, h] / w_max
            (g == h || w < 0.05) && continue
            lines!(ax1, [coords[g], coords[h]]; color=(:grey, 0.8w), linewidth=5w)
        end
        scatter!(ax1, coords; markersize=marker_size, color=:steelblue)
        ax2 = Axis(fig[row, 2]; xlabel="x (km)", title="$label deviation")
        return scatter!(
            ax2,
            coords;
            markersize=marker_size,
            color=delta,
            colormap=CairoMakie.Reverse(:RdBu),
            colorrange=(-lim, lim),
        )
    end
    fig = Figure(; size=(700, 640))
    maps_row!(fig, 1, K_true, delta_true, "True")
    sc = maps_row!(fig, 2, K_hat, delta_hat, "Posterior mean")
    Colorbar(fig[1:2, 3], sc; label="δ")
    fig
end
```

Patch reproduction numbers are recovered with the same helpers as in the single-patch model:

```@example renewal
R_patch = stack(r.R for r in patch_rets)
patch_true_rets = return_values(patch_model, patch_truth)

patch_grid() do ax, g
    ci_bands!(ax, view(R_patch, :, g, :))
    lines!(ax, patch_true_rets.R[:, g]; color=:firebrick)
end
```

## Where next

In practice model development loops.
You could explore extending the model with:

  - Delay uncertainty: the reporting delay is fixed at its posterior means; an informed prior from the delay posterior, or joint fitting by passing the linelist-conditioned `delay_model` into the composition in place of the fixed ``p_d``, would propagate it.
  - Right truncation and nowcasting, so recent partially reported days can be fitted rather than discarded, as in [epinowcast](https://package.epinowcast.org/).
  - A Gaussian process reproduction number model in place of the random walk, as in [EpiNow2](https://epiforecasts.io/EpiNow2/).
  - Additional data streams observed from the same infection process, for example wastewater concentrations as in [EpiSewer](https://github.com/adrian-lison/EpiSewer).

See [A Workflow for Infectious Disease Modelling](https://samabbott.co.uk/a-workflow-for-infectious-disease-modelling/paper.pdf) for a suggested workflow.
