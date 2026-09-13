# Shared harness for benchmarking alternative formulations of batched log densities under
# Reactant, on whichever device Reactant picks (GPU if available, otherwise CPU).
#
# Each per-distribution script defines a list of `VARIANTS`: `name => f` pairs where
# `f(params..., x)` computes the log density of `x` from array parameters. Wrap `f` in
# `ScalarIndexed` if it indexes arrays element by element inside a `@trace` loop: the
# harness then allows scalar indexing while compiling it, which would otherwise throw in a
# non-interactive session. (Indexing inside the trace becomes a slice op in the compiled
# program rather than a host round trip, so the warning is not relevant.) The harness
# compiles each variant's log density (the primal) and its gradient with respect to the
# parameters, checks that all variants agree with a plain CPU Enzyme evaluation of a
# reference variant, and reports the median time per evaluation of each across a range of
# sizes.
#
# The compiled gradient function returns the value and the gradients. Timing reads the
# scalar value back to the host, which waits for the whole computation to finish without
# copying the gradient arrays.
#
# `WREN_BENCH_NS` is a comma-separated list of N values (default 100,1000,10000,100000).

using Chairmarks: @be
using Enzyme: Enzyme, Reverse, Const, ReverseWithPrimal
using Printf: @printf, @sprintf
using Random: Xoshiro
using Reactant: Reactant, @trace
using Statistics: median

const NS = parse.(Int, split(get(ENV, "WREN_BENCH_NS", "100,1000,10000,100000"), ","))

function pretty_time(seconds)
    return if seconds < 1e-6
        @sprintf("%.3f ns", seconds * 1e9)
    elseif seconds < 1e-3
        @sprintf("%.3f μs", seconds * 1e6)
    elseif seconds < 1
        @sprintf("%.3f ms", seconds * 1e3)
    else
        @sprintf("%.3f s", seconds)
    end
end

# Gradient with respect to every parameter array; the observation `x` is constant.
struct ValueAndGrad{F}
    f::F
end
function (vg::ValueAndGrad)(params, x)
    result = Enzyme.gradient(ReverseWithPrimal, vg.f, params, Const(x))
    return result.val, first(result.derivs)
end

struct ScalarIndexed{F}
    f::F
end
_unwrap(f::ScalarIndexed) = f.f
_unwrap(f) = f
_compile_scope(::ScalarIndexed) = Reactant.allowscalar
_compile_scope(f) = (compile -> compile())

_host(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)
_host(x::Number) = x

# `make_inputs(rng, N)` returns `(params::Tuple, x)` for size `N`. `reference` is the
# variant that is also run under plain Enzyme on the CPU to check the others against.
function run_variants(variants, make_inputs, reference)
    println("Reactant device: ", Reactant.devices())
    println("Times are for one evaluation of the primal, and one of value-and-gradient\n")
    @printf("%-20s %9s %14s %14s\n", "variant", "N", "primal", "grad")
    for N in NS
        params, x = make_inputs(Xoshiro(468), N)
        ref_val, ref_grad = ValueAndGrad(reference)(params, x)
        params_ra = Reactant.to_rarray(params)
        x_ra = Reactant.to_rarray(x)
        for (name, variant) in variants
            f = _unwrap(variant)
            scope = _compile_scope(variant)
            primal = scope() do
                Reactant.@compile sync = true f(params_ra, x_ra)
            end
            isapprox(_host(primal(params_ra, x_ra)), ref_val; rtol=1e-6) ||
                @warn "primal value mismatch" name N
            primal_bench = @be _host(primal(params_ra, x_ra))

            vg = ValueAndGrad(f)
            grad_fn = scope() do
                Reactant.@compile sync = true vg(params_ra, x_ra)
            end
            val, grad = grad_fn(params_ra, x_ra)
            isapprox(_host(val), ref_val; rtol=1e-6) ||
                @warn "value mismatch" name N _host(val) ref_val
            for (g, rg) in zip(grad, ref_grad)
                isapprox(Array(g), rg; rtol=1e-5) || @warn "gradient mismatch" name N
            end
            grad_bench = @be _host(first(grad_fn(params_ra, x_ra)))

            @printf(
                "%-20s %9d %14s %14s\n",
                name,
                N,
                pretty_time(median(primal_bench).time),
                pretty_time(median(grad_bench).time),
            )
        end
        println()
    end
end
