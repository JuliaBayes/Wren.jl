# Shared harness for benchmarking alternative formulations of batched log densities, under
# plain Enzyme on the CPU and under Reactant on whichever device it picks (GPU if
# available, otherwise CPU).
#
# Each per-distribution script defines a list of `VARIANTS`: `name => f` pairs where
# `f(params..., x)` computes the log density of `x` from array parameters. Wrap `f` in
# `ScalarIndexed` if it indexes arrays element by element inside a `@trace` loop: the
# harness then allows scalar indexing while compiling it, which would otherwise throw in a
# non-interactive session. (Indexing inside the trace becomes a slice op in the compiled
# program rather than a host round trip, so the warning is not relevant.) The harness
# evaluates each variant's log density (the primal) and its gradient with respect to the
# parameters on each backend, checks that all variants agree with a plain CPU Enzyme
# evaluation of a reference variant, and reports the median time per evaluation of each
# across a range of sizes.
#
# Under Reactant, the compiled gradient function returns the value and the gradients.
# Timing reads the scalar value back to the host, as Wren's Reactant extension does on
# every call. This adds a fixed device-to-host round trip that is unrelated to the
# formulation, but it matches what a sampler currently pays.
#
# `WREN_BENCH_NS` is a comma-separated list of N values (default 100,1000,10000,100000,1000000).
# `WREN_BENCH_ELTYPE` is `Float64` (default) or `Float32`.
# `WREN_BENCH_BACKENDS` is a comma-separated subset of `enzyme,reactant`. By default both
# run, unless Reactant has a GPU, in which case only `reactant` runs so that GPU time is
# not spent on CPU benchmarks.

using Chairmarks: @be
using Enzyme: Enzyme, Reverse, Const, ReverseWithPrimal
using Pkg: Pkg
using Printf: @printf, @sprintf
using Random: Xoshiro
using Reactant: Reactant, @trace
using Statistics: median

const NS = parse.(Int, split(get(ENV, "WREN_BENCH_NS", "100,1000,10000,100000,1000000"), ","))
const ELTYPE = Dict("Float64" => Float64, "Float32" => Float32)[get(ENV, "WREN_BENCH_ELTYPE", "Float64")]
# Sums over up to 1e6 elements in Float32 differ noticeably between reduction orders.
const RTOL = max(1e-6, 1000 * eps(ELTYPE))

const SECONDS = 5.0

_on_gpu() = any(d -> Reactant.XLA.device_kind(d) != "cpu", Reactant.devices())
const BACKENDS = split(
    get(ENV, "WREN_BENCH_BACKENDS", _on_gpu() ? "reactant" : "enzyme,reactant"), ","
)

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

function _check(name, N, backend, val, grad, ref_val, ref_grad)
    isapprox(_host(val), ref_val; rtol=RTOL) ||
        @warn "value mismatch" backend name N _host(val) ref_val
    for (g, rg) in zip(grad, ref_grad)
        isapprox(Array(g), rg; rtol=RTOL) || @warn "gradient mismatch" backend name N
    end
end

# Each backend returns median (primal, grad) times for one variant.
function time_enzyme(name, N, variant, params, x, ref_val, ref_grad)
    f = _unwrap(variant)
    vg = ValueAndGrad(f)
    _check(name, N, "enzyme", f(params, x), last(vg(params, x)), ref_val, ref_grad)
    primal_bench = @be f(params, x) seconds=SECONDS
    grad_bench = @be vg(params, x) seconds=SECONDS
    return median(primal_bench).time, median(grad_bench).time
end

function time_reactant(name, N, variant, params, x, ref_val, ref_grad)
    f = _unwrap(variant)
    scope = _compile_scope(variant)
    params_ra = Reactant.to_rarray(params)
    x_ra = Reactant.to_rarray(x)
    vg = ValueAndGrad(f)
    primal = scope() do
        Reactant.@compile sync = true f(params_ra, x_ra)
    end
    grad_fn = scope() do
        Reactant.@compile sync = true vg(params_ra, x_ra)
    end
    _check(name, N, "reactant", primal(params_ra, x_ra), last(grad_fn(params_ra, x_ra)), ref_val, ref_grad)
    primal_bench = @be _host(primal(params_ra, x_ra)) seconds=SECONDS
        grad_bench = @be _host(first(grad_fn(params_ra, x_ra))) seconds=SECONDS
    return median(primal_bench).time, median(grad_bench).time
end

const TIMERS = Dict("enzyme" => time_enzyme, "reactant" => time_reactant)

# `make_inputs(rng, T, N)` returns `(params::Tuple, x)` of element type `T` for size `N`.
# `reference` is the
# variant that is run under plain Enzyme on the CPU to check the others against.
function print_status()
    deps = Pkg.dependencies()
    version_of(name) = string(only(d for d in values(deps) if d.name == name).version)
    println("Julia:            ", VERSION, " (", Threads.nthreads(), " threads)")
    println("CPU:              ", Sys.cpu_info()[1].model)
    println("Enzyme:           ", version_of("Enzyme"))
    println("Reactant:         ", version_of("Reactant"), " (Reactant_jll ", version_of("Reactant_jll"), ")")
    println("XLA platform:     ", Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    println("Reactant devices: ", join(Reactant.XLA.device_kind.(Reactant.devices()), ", "))
    # Reactant bundles its own CUDA, so the driver is the only host-side CUDA component.
    nvidia_smi = Sys.which("nvidia-smi")
    if !isnothing(nvidia_smi)
        query = "--query-gpu=name,driver_version,memory.total"
        println("nvidia-smi:       ", strip(read(`$nvidia_smi $query --format=csv,noheader`, String)))
    end
    println("Backends:         ", join(BACKENDS, ", "))
    println("Sizes:            ", join(NS, ", "))
    println("Element type:     ", ELTYPE)
    if "reactant" in BACKENDS
        # Fixed per-call costs on this device: dispatching a trivial compiled function,
        # and additionally reading one scalar back to the host.
        one = Reactant.to_rarray([1.0])
        trivial = Reactant.@compile sync = true sum(one)
        t_call = median(@be trivial(one) seconds=SECONDS).time
        t_host = median(@be _host(trivial(one)) seconds=SECONDS).time
        println("Reactant floor:   ", pretty_time(t_call), " per call, ", pretty_time(t_host), " with host readback")
    end
    println()
end

function run_variants(variants, make_inputs, reference)
    print_status()
    println("Times are for one evaluation of the primal, and one of value-and-gradient\n")
    @printf("%-20s %9s", "variant", "N")
    for backend in BACKENDS
        @printf(" %16s %16s", backend * " primal", backend * " grad")
    end
    println()
    for N in NS
        params, x = make_inputs(Xoshiro(468), ELTYPE, N)
        ref_val, ref_grad = ValueAndGrad(reference)(params, x)
        for (name, variant) in variants
            @printf("%-20s %9d", name, N)
            for backend in BACKENDS
                # A variant that a backend cannot handle is a result in itself, so record
                # it and carry on rather than aborting the whole run.
                times = try
                    TIMERS[backend](name, N, variant, params, x, ref_val, ref_grad)
                catch err
                    msg = first(split(sprint(showerror, err), '\n'))
                    @warn "$backend failed on $name at N=$N: $msg"
                    nothing
                end
                if isnothing(times)
                    @printf(" %16s %16s", "error", "error")
                else
                    @printf(" %16s %16s", pretty_time(times[1]), pretty_time(times[2]))
                end
            end
            println()
        end
        println()
    end
end
