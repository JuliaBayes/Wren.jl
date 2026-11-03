"""
    ensemble([rng], model, niters, nchains=1; nwalkers=nothing,
             move=StretchMove(), warmup=0, thin=1, adtype=nothing,
             initial_params=nothing, progress=true)

Sample a continuous, fixed-dimensional model with EnsembleMCMC.jl. Load
EnsembleMCMC to enable this sampler. Initial walkers are drawn from the model.
The default walker count is at least four and meets the chosen move's minimum.

Return one `VNChain` per walker, with `niters` rows and `nchains` independent
ensemble columns. Walkers within an ensemble interact and are not independent chains.
`warmup` discards sweeps; `thin` retains every nth sweep. Moves do not adapt.

For custom starts, pass complete model-space `VarNamedTuple` values as
`initial_params[walker, ensemble]`, or a vector for one ensemble.

Pass `adtype=AutoReactant(AutoEnzyme(; mode=Reverse))` to use Reactant for model
evaluation on its selected device. Ensemble moves and stored samples stay on the CPU.
"""
function ensemble end
