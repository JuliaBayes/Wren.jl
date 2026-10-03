using WrenCore, Distributions, Test

@testset "#40 short function definitions" begin
    @wren short_def(a) = x ~ Normal(a)
    accs = rand_with_logp(short_def(1.0))
    @test logprior(accs) ≈ logpdf(Normal(1.0), parameters(accs)[@vn(x)])
end
