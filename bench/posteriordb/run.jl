using Wren: Wren, VecModel, lp, lp_and_grad, benchmark_model
using DynamicPPL: DynamicPPL
using ADTypes: AutoEnzyme, AutoMooncake, AutoReactant
using Enzyme: Reverse, set_runtime_activity
using Mooncake: Mooncake
using Reactant: Reactant
using Printf: @sprintf

# Each entry is either a time in seconds, or an exception that was caught.
const BenchEntry = Union{Real,Exception}

struct BenchResult
    name::String
    # Primals
    wren_primal::BenchEntry
    turing_primal::BenchEntry
    # Gradients
    wren_enzyme::BenchEntry
    wren_mooncake::BenchEntry
    turing_enzyme::BenchEntry
    turing_mooncake::BenchEntry
    # Reactant
    wren_reactant_primal::BenchEntry
    wren_reactant_grad::BenchEntry
end

function _try(f)
    try
        return f()
    catch e
        return e
    end
end

function _format(x::Exception, base)
    msg = join(Base.split(sprint(showerror, x), "\n"), "; ")
    # Truncate long messages
    MAX_LEN = 80
    if length(msg) > MAX_LEN
        msg = msg[1:MAX_LEN] * "…"
    end
    return "ERR: $msg"
end
# `base` is `nothing` if there is nothing to compare against, or an exception if the
# baseline failed; in both cases the ratio is left out.
function _format(x::Real, base::Union{BenchEntry,Nothing})
    ratio = base isa Real ? @sprintf("(%.2fx)", x / base) : ""
    absolute = if x < 1e-6
        @sprintf("%.3f ns", x * 1e9)
    elseif x < 1e-3
        @sprintf("%.3f μs", x * 1e6)
    elseif x < 1.0
        @sprintf("%.3f ms", x * 1e3)
    else
        @sprintf("%.3f s", x)
    end
    return lpad(absolute, 11) * " " * ratio
end

function Base.show(io::IO, ::MIME"text/plain", r::BenchResult)
    println(io, "═══ ", r.name, " ═══")
    println(io, "  Primals:")
    println(io, "    Wren           : ", _format(r.wren_primal, nothing))
    println(io, "    Turing         : ", _format(r.turing_primal, r.wren_primal))
    println(io, "    Wren/Reactant  : ", _format(r.wren_reactant_primal, r.wren_primal))
    println(io, "  Gradients:")
    println(io, "    Wren+Enzyme    : ", _format(r.wren_enzyme, nothing))
    println(io, "    Wren+Mooncake  : ", _format(r.wren_mooncake, r.wren_enzyme))
    println(io, "    Wren+Reactant  : ", _format(r.wren_reactant_grad, r.wren_enzyme))
    println(io, "    Turing+Enzyme  : ", _format(r.turing_enzyme, r.wren_enzyme))
    println(io, "    Turing+Mooncake: ", _format(r.turing_mooncake, r.wren_enzyme))
end

function bench(mod)
    name = string(nameof(mod))
    wren_model = mod.wren_model
    turing_model = mod.turing_model

    enzyme = AutoEnzyme(; mode=Reverse)
    mooncake = AutoMooncake(; config=nothing)
    reactant = AutoReactant(enzyme)

    # -- Wren primals + gradients --
    wren_enzyme_result = _try() do
        benchmark_model(wren_model, enzyme; verbose=false)
    end
    wren_primal =
        wren_enzyme_result isa Exception ? wren_enzyme_result :
        wren_enzyme_result.lp_bench.time
    wren_enzyme =
        wren_enzyme_result isa Exception ? wren_enzyme_result :
        wren_enzyme_result.lp_and_grad_bench.time
    wren_mooncake = _try() do
        r = benchmark_model(wren_model, mooncake; verbose=false)
        r.lp_and_grad_bench.time
    end

    # -- Turing (DynamicPPL) --
    run_ad = DynamicPPL.TestUtils.AD.run_ad
    turing_enzyme_adtype = AutoEnzyme(; mode=set_runtime_activity(Reverse))
    turing_enzyme_result = _try() do
        run_ad(turing_model, turing_enzyme_adtype; test=false, benchmark=true, verbose=false)
    end
    turing_primal =
        turing_enzyme_result isa Exception ? turing_enzyme_result :
        turing_enzyme_result.primal_time
    turing_enzyme =
        turing_enzyme_result isa Exception ? turing_enzyme_result :
        turing_enzyme_result.grad_time
    turing_mooncake = _try() do
        r = run_ad(turing_model, mooncake; test=false, benchmark=true, verbose=false)
        r.grad_time
    end

    # -- Wren + Reactant --
    wren_reactant_primal = _try() do
        # Benchmarked separately from the gradient, so that we still get a primal time
        # even if the gradient fails.
        vecmodel = VecModel(wren_model, reactant)
        params = vecmodel.sample
        # Timed the same way as in `benchmark_model`.
        Wren.WrenCore._median_bench(() -> lp(vecmodel, params), 1).time
    end
    wren_reactant_grad = _try() do
        r = benchmark_model(wren_model, reactant; verbose=false)
        r.lp_and_grad_bench.time
    end

    return BenchResult(
        name,
        wren_primal,
        turing_primal,
        wren_enzyme,
        wren_mooncake,
        turing_enzyme,
        turing_mooncake,
        wren_reactant_primal,
        wren_reactant_grad,
    )
end

include("models/ark.jl")
include("models/eight_schools_centered.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/kidscore_interaction.jl")
include("models/mh.jl")
include("models/rate1.jl")
include("models/sblri_blr.jl")

for mod in (
    ArK,
    EightSchoolsCentered,
    GLMBinomial,
    GLMPoisson,
    GLMMPoisson,
    GPRegr,
    KidscoreInteraction,
    Mh,
    Rate1,
    SblriBLR,
)
    result = bench(mod)
    show(stdout, MIME"text/plain"(), result)
    println()
end
