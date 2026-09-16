using WrenDists, Enzyme, Chairmarks
using WrenDists: logdensityof, BatchedLogNormal, log2π
using Statistics: median

struct BroadcastLogNormal{M,S}
    mu::M
    sigma::S
end
function logdensityof_broadcast(d::BroadcastLogNormal, x)
    mu, sigma = d.mu, d.sigma
    lognorm = length(x) * convert(float(eltype(x)), log2π) / 2
    sum(@. -(abs2((log(x) - mu) / sigma)) / 2 - log(sigma) - log(x)) - lognorm
end

N = 1_000_000
mu = randn(N);
sigma = exp.(0.3 .* randn(N))
x = exp.(mu .+ sigma .* randn(N))

d_wd = BatchedLogNormal(mu, sigma)
d_bc = BroadcastLogNormal(mu, sigma)

println("=== Primal (N=$N) ===")
t1p = @be logdensityof($d_wd, $x) seconds=5.0
t2p = @be logdensityof_broadcast($d_bc, $x) seconds=5.0
println("  WrenDists (loop+rule): $(median(t1p))")
println("  broadcast-only:        $(median(t2p))")

println("\n=== Enzyme grad (N=$N) ===")
t1g = @be Enzyme.gradient(Reverse, logdensityof, $d_wd, Const($x)) seconds=5.0
t2g = @be Enzyme.gradient(Reverse, logdensityof_broadcast, $d_bc, Const($x)) seconds=5.0
println("  WrenDists (loop+rule): $(median(t1g))")
println("  broadcast-only:        $(median(t2g))")
