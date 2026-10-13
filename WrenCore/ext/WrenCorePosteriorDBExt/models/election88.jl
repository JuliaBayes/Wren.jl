# PosteriorDB: election88-election88_full
module Election88

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "election88-election88_full",
    )))
end

@wren function election88(age, edu, age_edu, state, region, black, female,
    v_prev_full, y, n_age, n_edu, n_age_edu, n_state, n_region)
    @iid beta ~ WD.Normal(0.0, 100.0) size=5
    raw_sigma_a ~ WD.Flat()
    raw_sigma_b ~ WD.Flat()
    raw_sigma_c ~ WD.Flat()
    raw_sigma_d ~ WD.Flat()
    raw_sigma_e ~ WD.Flat()
    ratio_a = inv(1 + exp(-raw_sigma_a))
    ratio_b = inv(1 + exp(-raw_sigma_b))
    ratio_c = inv(1 + exp(-raw_sigma_c))
    ratio_d = inv(1 + exp(-raw_sigma_d))
    ratio_e = inv(1 + exp(-raw_sigma_e))
    # Stan's five (0, 100) group scales have no prior factors.
    @addloglikelihood 5 * log(100.0) +
        log(ratio_a) + log1p(-ratio_a) +
        log(ratio_b) + log1p(-ratio_b) +
        log(ratio_c) + log1p(-ratio_c) +
        log(ratio_d) + log1p(-ratio_d) +
        log(ratio_e) + log1p(-ratio_e)
    @iid a ~ WD.Normal(0.0, 100 * ratio_a) size=n_age
    @iid b ~ WD.Normal(0.0, 100 * ratio_b) size=n_edu
    @iid c ~ WD.Normal(0.0, 100 * ratio_c) size=n_age_edu
    @iid d ~ WD.Normal(0.0, 100 * ratio_d) size=n_state
    @iid e ~ WD.Normal(0.0, 100 * ratio_e) size=n_region
    eta = beta[1] .+ beta[2] .* black .+ beta[3] .* female .+
        beta[5] .* (female .* black) .+ beta[4] .* v_prev_full .+
        a[age] .+ b[edu] .+ c[age_edu] .+ d[state] .+ e[region]
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.BernoulliLogit(eta[i])
    end
    return nothing
end

const wren_model = election88(data["age"], data["edu"], data["age_edu"],
    data["state"], data["region_full"], data["black"], data["female"],
    data["v_prev_full"], data["y"], data["n_age"], data["n_edu"],
    data["n_age_edu"], data["n_state"], data["n_region_full"])

end
