module WrenCoreOptimizationExt

using WrenCore
import DifferentiationInterface as DI
import Optimization
using VarNames: densify!!

function WrenCore.optimize(
    model::Model,
    solver;
    adtype::DI.AbstractADType=DEFAULT_AD,
    init=FromPrior(),
    lp_function=logjoint,
    reduce_copies=true,
    kwargs...,
)
    # The starting point is set by `init`, so don't let users pass Optimization.jl's `u0`.
    haskey(kwargs, :u0) &&
        throw(ArgumentError("Use `init` instead of `u0` to set the starting point."))
    init = WrenCore.to_initialization(init)
    density = VecModel(
        model,
        adtype;
        latent_source=WrenCore._initial_source(init),
        lp_function,
        reduce_copies,
    )
    fg = function (gradient, x, p)
        gradient === nothing && return -lp(density, x)
        value, _ = lp_and_grad!(gradient, density, x)
        gradient .*= -1
        return -value
    end
    objective = Optimization.OptimizationFunction(
        (x, p) -> -lp(density, x);
        grad=(g, x, p) -> (fg(g, x, p); nothing),
        fg,
    )
    problem = Optimization.OptimizationProblem(
        objective,
        WrenCore._initial_position(density, init),
    )
    solution = Optimization.solve(problem, solver; kwargs...)
    params = densify!!(parameters(density, solution.u))
    return PointEstimate(lp_function, params, -solution.objective, density, solution)
end

end
