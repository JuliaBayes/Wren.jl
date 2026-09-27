using Pkg
Pkg.add("Reactant")
using Reactant

function test_halft_quantile_reactant(d, us)
    u_ra = Reactant.to_rarray(first(us); track_numbers=true)
    f = Reactant.@compile WrenDists._halft_upper_quantile(d, u_ra)
    for u in us
        u_ra = Reactant.to_rarray(u; track_numbers=true)
        @test Float64(f(d, u_ra)) ≈ WrenDists._halft_upper_quantile(d, u) rtol = 1e-10
    end
end
