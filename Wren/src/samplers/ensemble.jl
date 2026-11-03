"""
    ensemble(
        [rng::Random.AbstractRNG=default_rng()], model::Model, niters::Integer,
        nchains::Integer=1; nwalkers=nothing, move=StretchMove(), warmup=0,
        thin=1, initial_params=nothing, progress=true,
    ) -> Vector{<:FlexiChains.VNChain}

Sample a fixed-dimensional continuous Wren model with EnsembleMCMC.jl. Load EnsembleMCMC to enable
this optional sampler. No gradients are needed. Model evaluation is serial.

`nchains` counts independent ensembles. Each ensemble contains `nwalkers` coupled
walkers. The default walker count is at least four and meets the selected move's
minimum. `niters` counts retained complete sweeps. Discard `warmup` sweeps, then
retain every `thin`-th sweep. No adaptation or convergence test runs automatically.

Return one `VNChain` per walker, in walker order. Each chain has `niters` rows and
`nchains` independent ensemble columns. Use existing summaries and diagnostics on
these individual chains. The returned walker chains are mutually dependent: do
not concatenate them into independent chains or sum their ESS values.

Initialization uses the model distributions' sampling methods and the supplied
RNG. Alternatively, pass complete model-space `VarNamedTuple` values in an
`initial_params` matrix with axes (walker, ensemble). A vector is accepted for one
ensemble. The walker count is inferred from explicit values when omitted. Initial
positions must span the unconstrained parameter space and have finite log density.
Invalid starts fail; supply explicit values when the model's sampling methods do
not provide usable starts.

Each draw contains model-space parameters and log probabilities. Its extras also
include transformed `log_density`, `accepted`, `move_index`, `walker_id`, and
`sweep`. Acceptance and move metadata describe the retained sweep only, including
when `thin > 1`. Rejected steps repeat the previous position.

All initialization and transition randomness comes from `rng`, which advances.
Equal seeds replay a run in a fixed environment. `progress` reports completion of
each ensemble. Set it to `false` to suppress output.
"""
function ensemble end
