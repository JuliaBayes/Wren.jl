using Pkg
Pkg.add("Reactant")
using Reactant

function test_reactant(d, points)
    x_ra = Reactant.to_rarray(first(points))
    f = Reactant.@compile logdensityof(d, x_ra)
    for x in points
        x_ra = Reactant.to_rarray(x)
        @test Float64(f(d, x_ra)) ≈ logdensityof(d, x)
    end
end
