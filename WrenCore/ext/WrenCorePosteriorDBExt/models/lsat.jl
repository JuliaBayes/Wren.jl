# PosteriorDB: lsat_data-lsat_model
module LSAT

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "lsat_data-lsat_model")))
end

function expanded_responses(data)
    T, N = data["T"], data["N"]
    r = Matrix{Int}(undef, T, N)
    previous = 0
    for pattern in 1:data["R"]
        for student in (previous + 1):data["culm"][pattern]
            for question in 1:T
                r[question, student] = data["response"][pattern, question]
            end
        end
        previous = data["culm"][pattern]
    end
    previous == N || throw(ArgumentError("LSAT response counts do not sum to N"))
    return r
end

function lsat_logp(r, alpha, theta, beta)
    total = zero(alpha[1] + theta[1] + beta)
    for student in axes(r, 2), question in axes(r, 1)
        eta = beta * theta[student] - alpha[question]
        total += logdensityof(WD.BernoulliLogit(eta), r[question, student])
    end
    return total
end

@wren function lsat(r, T, N)
    @iid alpha ~ WD.Normal(0.0, 100.0) size=T
    @iid theta ~ WD.Normal(0.0, 1.0) size=N
    # Stan's positive Normal factor differs by the constant log(2).
    beta ~ WD.HalfNormal(0.0, 100.0)
    @addloglikelihood lsat_logp(r, alpha, theta, beta)
    return nothing
end

const wren_model = lsat(expanded_responses(data), data["T"], data["N"])

end
