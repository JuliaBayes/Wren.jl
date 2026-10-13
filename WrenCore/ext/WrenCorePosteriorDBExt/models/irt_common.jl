module IRTCommon

using Statistics: mean, std
using WrenCore: logdensityof
using WrenDists: WrenDists as WD

function adjusted_covariates(W)
    J, K = size(W)
    W_adj = Matrix{Float64}(undef, J, K)
    W_adj[:, 1] .= W[:, 1]
    for k in 2:K
        column = view(W, :, k)
        lo, hi = extrema(column)
        scale = all(x -> x == lo || x == hi, column) ? hi - lo : 2 * std(column)
        center = mean(column)
        W_adj[:, k] .= (column .- center) ./ scale
    end
    return W_adj
end

function beta_theta_logdensity(W_adj, beta_free, theta, lambda_adj)
    total = zero(beta_free[1] + theta[1] + lambda_adj[1])
    beta_last = -sum(beta_free)
    for b in beta_free
        total += logdensityof(WD.Normal(0.0, 3.0), b)
    end
    total += logdensityof(WD.Normal(0.0, 3.0), beta_last)
    for j in eachindex(theta)
        location = zero(total)
        for k in eachindex(lambda_adj)
            location += W_adj[j, k] * lambda_adj[k]
        end
        total += logdensityof(WD.Normal(location, 1.0), theta[j])
    end
    return total
end

function graded_logdensity(y, eta, thresholds, offset, first, m)
    score = zero(eta + offset + thresholds[first])
    selected = score
    lognormalizer = score
    for r in 1:m
        score += eta - offset - thresholds[first + r - 1]
        selected = y == r ? score : selected
        lognormalizer = logaddexp(lognormalizer, score)
    end
    return selected - lognormalizer
end

logaddexp(a, b) = a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))

end
