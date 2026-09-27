# TODO(penelopeysm): use DIT.Scenario for this when the following is fixed:
# https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/1069
struct ADTestCase{F,X,C,S}
    f::F
    x::X
    contexts::C
    skipped_adtypes::S
end

function test_dist end

test_dist_reactant(d, points) = nothing
