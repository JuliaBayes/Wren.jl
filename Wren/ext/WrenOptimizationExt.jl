module WrenOptimizationExt

using Wren:
    Wren,
    Model,
    DEFAULT_AD,
    FromPrior,
    logdensity,
    logjoint,
    lp,
    lp_and_grad!,
    Draw,
    parameters
import DifferentiationInterface as DI
import Optimization

function Wren.optimize(
    model::Model,
    solver;
    adtype::DI.AbstractADType=DEFAULT_AD,
    latent_source=FromPrior(),
    lp_function=logjoint,
    reduce_copies=true,
    kwargs...,
)
    density = logdensity(model, adtype; latent_source, lp_function, reduce_copies)
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
    problem = Optimization.OptimizationProblem(objective, copy(density.sample))
    solution = Optimization.solve(problem, solver; kwargs...)
    params = parameters(Draw(solution.u, density; include_log_probs=false))
    return (; params, solution)
end

end
