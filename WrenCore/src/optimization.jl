"""
    PointEstimate

The result of [`optimize`](@ref), [`MAP`](@ref), or [`MLE`](@ref).

## Fields

- `lp_function`: the objective that was maximised, e.g. `logjoint` for MAP or
  `loglikelihood` for MLE.

- `parameters`: a `VarNamedTuple` of the fitted parameters, in model space (i.e. not in
  unconstrained coordinates, even though the optimisation itself is carried out in
  unconstrained coordinates). To access this we recommend using `parameters(::PointEstimate)`
  rather than accessing the field, but both are equivalent.

- `lp`: the value of `lp_function` at `parameters`. This is the actual log-density, not the
  negated value that the solver minimises.

- `vecmodel`: the [`VecModel`](@ref) used to evaluate the objective. It should hold that
  `p.lp ≈ lp(p.vecmodel, p.solution.u)` for a `p::PointEstimate`.

- `solution`: the solution object returned by `Optimization.solve`. Check
  `p.solution.retcode` to see whether the solver converged.
"""
struct PointEstimate{F,P<:VarNamedTuple,LP<:Real,V<:VecModel,S}
    lp_function::F
    parameters::P
    lp::LP
    vecmodel::V
    solution::S
end

"""
    parameters(p::PointEstimate)::VarNamedTuple

Return the fitted model-space parameters of `p` as a `VarNamedTuple`.
"""
VarNames.parameters(p::PointEstimate) = p.parameters

function Base.show(io::IO, ::MIME"text/plain", p::PointEstimate)
    println(io, "PointEstimate")
    print(io, "├─ ")
    printstyled(io, "lp_function"; color=:cyan)
    print(io, " => ")
    p.lp_function isa Function ? print(io, nameof(p.lp_function)) : show(io, p.lp_function)
    println(io)
    print(io, "├─ ")
    printstyled(io, "lp"; color=:cyan)
    print(io, " => ")
    show(io, p.lp)
    println(io)
    print(io, "├─ ")
    printstyled(io, "parameters"; color=:cyan)
    print(io, " => ")
    if isempty(p.parameters)
        print(io, "VarNamedTuple()")
    else
        vnt_pretty_print(io, p.parameters, "│  ", 0)
    end
    println(io)
    # These fields are large, so only show an abbreviated type.
    print(io, "├─ ")
    printstyled(io, "vecmodel"; color=:cyan)
    print(io, "::", _abbreviated_type(p.vecmodel))
    println(io)
    print(io, "└─ ")
    printstyled(io, "solution"; color=:cyan)
    print(io, "::", _abbreviated_type(p.solution))
    println(io)
    # The return code is the most useful part of the solution, so show it underneath.
    print(io, "   └─ ")
    printstyled(io, "retcode"; color=:cyan)
    print(io, " => ")
    print(io, p.solution.retcode)
    return nothing
end

function _abbreviated_type(x)
    T = typeof(x)
    name = string(nameof(T))
    return T isa DataType && !isempty(T.parameters) ? name * "{…}" : name
end

"""
    optimize(
        model,
        solver;
        adtype=DEFAULT_AD,
        init=FromPrior(),
        lp_function=logjoint,
        reduce_copies=true,
        solve_kwargs...
    )

Find a maximum a posteriori (MAP) estimate with an Optimization.jl solver. Requires
Optimization.jl and the selected AD backend to be loaded.

Returns a [`PointEstimate`](@ref), which contains the fitted model-space parameters, the
value of the objective there, and the native Optimization solution (including its
convergence status). The solver minimises the negative original-space log-density in
unconstrained coordinates.

The starting point is chosen by `init`, which should be an [`AbstractInitialization`](@ref)
or anything that can be implicitly converted to one. The default is to attempt drawing from
the prior.

Other keywords are forwarded to `Optimization.solve`, although passing `u0` is not allowed
(you should use `init` instead).
"""
function optimize end

"""
    MAP(model, solver; kwargs...)

Find the original-space maximum a posteriori estimate of a model.

All arguments are forwarded to [`optimize`](@ref) but with the objective function set to
`logjoint`. Please see its documentation for details.
"""
MAP(model::Model, solver; kwargs...) =
    optimize(model, solver; kwargs..., lp_function=logjoint)

"""
    MLE(model, solver; kwargs...)

Find the original-space maximum likelihood estimate of a model.

All arguments are forwarded to [`optimize`](@ref) but with the objective function set to
`logjoint`. Please see its documentation for details.
"""
MLE(model::Model, solver; kwargs...) =
    optimize(model, solver; kwargs..., lp_function=loglikelihood)
