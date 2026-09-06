#!/usr/bin/env julia
#
# Benchmark logp + gradient evaluation for Wren vs Stan on PosteriorDB models.
#
# Usage:
#   julia --project=benchmark_wren benchmark_wren/posteriordb.jl                    # all models
#   julia --project=benchmark_wren benchmark_wren/posteriordb.jl sblri-blr          # one model
#   julia --project=benchmark_wren benchmark_wren/posteriordb.jl --logdensity-only  # skip gradients
#   julia --project=benchmark_wren benchmark_wren/posteriordb.jl --stan-only        # skip Wren
#   julia --project=benchmark_wren benchmark_wren/posteriordb.jl --wren-only        # skip Stan
#   julia --project=benchmark_wren benchmark_wren/posteriordb.jl --verify sblri-blr
#
# Stan and Wren are benchmarked at separate reproducible draws, so the timings are
# comparable but the log-density values are not. Several priors are substituted in
# `posteriordb_models.jl`, which changes the posterior as well.

using Base64: base64decode, base64encode
using Printf, Random, LinearAlgebra
using SHA: sha256
using Chairmarks: @be
using Statistics: median
using ADTypes: AutoEnzyme, AutoReactant
import Enzyme
using Enzyme: Reverse
import BridgeStan
import PosteriorDB
import ProbabilityMeasures
import Reactant
import Wren

if haskey(ENV, "BRIDGESTAN_PATH")
    BridgeStan.set_bridgestan_path!(ENV["BRIDGESTAN_PATH"])
end

# ── PosteriorDB setup ──

const PDB = PosteriorDB.database()

# ── Helpers ──

function fmt_time(t)
    t === nothing && return "err"
    isnan(t) && return "err"
    isinf(t) && return "err"
    if t < 1e-6
        @sprintf("%.1f ns", t * 1e9)
    elseif t < 1e-3
        @sprintf("%.1f μs", t * 1e6)
    elseif t < 1.0
        @sprintf("%.2f ms", t * 1e3)
    else
        @sprintf("%.3f s", t)
    end
end

function fmt_ratio(numerator, denominator)
    ratio = numerator / denominator
    return isfinite(ratio) ? @sprintf("%.2f×", ratio) : "err"
end

function cpu_name()
    if Sys.islinux() && isfile("/proc/cpuinfo")
        info = read("/proc/cpuinfo", String)
        names = unique(
            strip(only(match.captures)) for
            match in eachmatch(r"(?m)^model name\s*:\s*(.+)$", info)
        )
        isempty(names) || return join(names, " + ")
    end
    return Sys.CPU_NAME
end

struct BenchmarkManifest
    content::String
    sha256::String

    function BenchmarkManifest(content::AbstractString)
        value = String(content)
        return new(value, bytes2hex(sha256(value)))
    end
end

function Base.:(==)(left::BenchmarkManifest, right::BenchmarkManifest)
    return left.sha256 == right.sha256 && left.content == right.content
end
Base.isequal(left::BenchmarkManifest, right::BenchmarkManifest) = left == right
function Base.hash(manifest::BenchmarkManifest, h::UInt)
    return hash(manifest.content, hash(manifest.sha256, h))
end

function encode_manifest(manifest::BenchmarkManifest)
    return "$(manifest.sha256):$(base64encode(manifest.content))"
end

encode_metadata(value) = value
encode_metadata(manifest::BenchmarkManifest) = encode_manifest(manifest)

function decode_manifest(value)
    fields = Base.split(value, ':'; limit=2)
    length(fields) == 2 || error("malformed benchmark manifest")
    expected_sha256, encoded = fields
    manifest = BenchmarkManifest(String(base64decode(encoded)))
    manifest.sha256 == expected_sha256 || error("benchmark manifest checksum mismatch")
    return manifest
end

function benchmark_manifest()
    project = Base.active_project()
    project === nothing && error("no active benchmark project")
    path = joinpath(dirname(project), "Manifest.toml")
    isfile(path) || error("benchmark manifest does not exist: $path")
    return BenchmarkManifest(read(path, String))
end

"""
    source_revision(package)

The short commit of `package`'s checkout, suffixed with `-dirty` when the tracked
files differ from it. Wren and ProbabilityMeasures are benchmarked while they are
being developed, so an unclean tree is recorded rather than rejected.
"""
function source_revision(package)
    path = dirname(dirname(pathof(package)))
    root = try
        readchomp(pipeline(`git -C $path rev-parse --show-toplevel`; stderr=devnull))
    catch
        return "manifest tree"
    end
    commit = readchomp(`git -C $root rev-parse --short HEAD`)
    status = read(`git -C $root status --porcelain --untracked-files=no`, String)
    return isempty(status) ? commit : "$commit-dirty"
end

struct RunMetadata
    julia_version::String
    system::String
    cpu::String
    cpu_threads::Int
    memory_gib::Float64
    julia_threads::Int
    blas_vendor::String
    blas_threads::Int
    wren_version::String
    reactant_version::String
    enzyme_version::String
    wren_commit::String
    probabilitymeasures_commit::String
    manifest::BenchmarkManifest
    bridgestan_version::String
    stan_version::String
    logdensity_only::Bool
    stan_only::Bool
    wren_only::Bool
    shard_count::Int
    shard_index::Int
    seed::Int
end

function run_metadata(
    stan_version;
    logdensity_only,
    stan_only,
    wren_only,
    shard_count=1,
    shard_index=1,
    seed=468,
    wren_commit=source_revision(Wren),
    probabilitymeasures_commit=source_revision(ProbabilityMeasures),
    manifest=benchmark_manifest(),
)
    return RunMetadata(
        string(VERSION),
        "$(Sys.KERNEL) $(Sys.ARCH)",
        cpu_name(),
        Sys.CPU_THREADS,
        Sys.total_memory() / 2.0^30,
        Threads.nthreads(),
        string(LinearAlgebra.BLAS.vendor()),
        LinearAlgebra.BLAS.get_num_threads(),
        string(pkgversion(Wren)),
        string(pkgversion(Reactant)),
        string(pkgversion(Enzyme)),
        wren_commit,
        probabilitymeasures_commit,
        manifest,
        string(pkgversion(BridgeStan)),
        stan_version,
        logdensity_only,
        stan_only,
        wren_only,
        shard_count,
        shard_index,
        seed,
    )
end

function write_environment(io, metadata::RunMetadata)
    println(io, "- Julia: $(metadata.julia_version)")
    println(io, "- System: $(metadata.system)")
    println(io, "- CPU: $(metadata.cpu) ($(metadata.cpu_threads) logical threads)")
    println(io, "- Memory: $(@sprintf("%.1f", metadata.memory_gib)) GiB")
    println(
        io,
        "- Benchmark parallelism: $(metadata.shard_count) Julia process(es), $(metadata.julia_threads) Julia thread(s) each",
    )
    println(io, "- BLAS: $(metadata.blas_vendor), $(metadata.blas_threads) thread(s)")
    println(
        io,
        "- Source revisions: Wren `$(metadata.wren_commit)`, ProbabilityMeasures `$(metadata.probabilitymeasures_commit)`",
    )
    return println(
        io,
        "- Packages: Wren $(metadata.wren_version), Reactant $(metadata.reactant_version), Enzyme $(metadata.enzyme_version), BridgeStan $(metadata.bridgestan_version), Stan $(metadata.stan_version)",
    )
end

median_time(bench) = median(bench).time

const ENZYME_BACKEND = AutoEnzyme(; mode=Reverse)
const REACTANT_BACKEND = AutoReactant(AutoEnzyme(; mode=Reverse))

# ── Discover models ──

include(joinpath(@__DIR__, "posteriordb_models.jl"))

function vec_model(model, adtype, seed)
    return Wren.VecModel(model, adtype; latent_source=Wren.FromPrior(Xoshiro(seed)))
end

"""
    unconstrained_point(vecmodel, seed; scale=0.2)

A reproducible point of `vecmodel`'s unconstrained space at which the log density
is finite, drawn by the same rule as the Stan point in [`random_valid_points`](@ref).

Drawing from the prior instead would put the vague-prior models many orders of
magnitude away from the data.
"""
function unconstrained_point(vecmodel, seed; scale=0.2)
    rng = Xoshiro(seed)
    for _ in 1:100
        q = scale .* randn(rng, vecmodel.dimension)
        isfinite(Wren.lp(vecmodel, q)) && return q
    end
    return error("could not draw a valid unconstrained parameter vector")
end

"""
    bench_wren(model, model_name, seed; logdensity_only=false)

Time Wren's log density under plain Julia and under Reactant, and its gradient
under Enzyme and under Reactant, all at one point drawn from `seed`.
"""
function bench_wren(model, model_name, seed; logdensity_only=false)
    plain = vec_model(model, nothing, seed)
    params = unconstrained_point(plain, seed)
    Wren.lp(plain, params)
    primal = median_time(@be Wren.lp($plain, $params))

    reactant_primal = NaN
    reactant_grad = NaN
    try
        reactant = vec_model(model, REACTANT_BACKEND, seed)
        reactant_params = convert(typeof(reactant.sample), params)
        Wren.lp(reactant, reactant_params)
        reactant_primal = median_time(@be Wren.lp($reactant, $reactant_params))
        if !logdensity_only
            Wren.lp_and_grad(reactant, reactant_params)
            reactant_grad = median_time(@be Wren.lp_and_grad($reactant, $reactant_params))
        end
    catch exception
        @warn "$model_name: Reactant failed — $(typeof(exception))"
    end

    enzyme_grad = NaN
    if !logdensity_only
        try
            enzyme = vec_model(model, ENZYME_BACKEND, seed)
            Wren.lp_and_grad(enzyme, params)
            enzyme_grad = median_time(@be Wren.lp_and_grad($enzyme, $params))
        catch exception
            @warn "$model_name: Enzyme failed — $(typeof(exception))"
        end
    end

    return (; primal, reactant_primal, enzyme_grad, reactant_grad, dim=plain.dimension)
end

# ── Reference results for Turing ──
#
# `benchmarks/posteriordb.md` records DynamicPPL and Stan timings from a
# different machine. Absolute times do not carry across machines but ratios
# against Stan do, so the report shows the recorded Turing/Stan and
# Mooncake/Stan ratios beside the ratios measured here.

const REFERENCE_MARKDOWN = joinpath(@__DIR__, "..", "benchmarks", "posteriordb.md")

const TIME_UNITS = Dict("ns" => 1e-9, "μs" => 1e-6, "ms" => 1e-3, "s" => 1.0)

function parse_reference_time(field)
    parts = Base.split(field)
    length(parts) == 2 || return NaN
    unit = get(TIME_UNITS, parts[2], nothing)
    unit === nothing && return NaN
    return parse(Float64, parts[1]) * unit
end

struct ReferenceRow
    turing_primal::Float64
    stan_primal::Float64
    mooncake_grad::Float64
    stan_grad::Float64
end

"""
    read_reference(path)

The `Model dim Turing Stan FwdDiff EzyRvs McRvs Stan` rows of a DynamicPPL
PosteriorDB report, keyed by posterior name. Returns an empty dictionary when the
report is absent.
"""
function read_reference(path)
    rows = Dict{String,ReferenceRow}()
    isfile(path) || return rows
    for line in eachline(path)
        fields = Base.split(strip(line), r" {2,}")
        length(fields) == 9 || continue
        all(!isempty, fields) || continue
        tryparse(Int, fields[2]) === nothing && continue
        rows[fields[1]] = ReferenceRow(
            parse_reference_time(fields[3]),
            parse_reference_time(fields[4]),
            parse_reference_time(fields[7]),
            parse_reference_time(fields[8]),
        )
    end
    return rows
end

# ── Results ──

struct Result
    name::String
    dim::Int
    wren_primal::Float64
    wren_reactant_primal::Float64
    stan_primal::Float64
    wren_enzyme_grad::Float64
    wren_reactant_grad::Float64
    stan_grad::Float64
end

const RESULT_COLUMNS = (
    "model",
    "dim",
    "wren_primal_s",
    "wren_reactant_primal_s",
    "stan_primal_s",
    "enzyme_gradient_s",
    "reactant_gradient_s",
    "stan_gradient_s",
)
const CHECKPOINT_FORMAT = "1"

function write_result(io, result)
    return println(
        io,
        join(
            (
                result.name,
                result.dim,
                result.wren_primal,
                result.wren_reactant_primal,
                result.stan_primal,
                result.wren_enzyme_grad,
                result.wren_reactant_grad,
                result.stan_grad,
            ),
            '\t',
        ),
    )
end

function write_checkpoint_header(io, metadata::RunMetadata)
    println(io, "# posteriordb_format\t$CHECKPOINT_FORMAT")
    for name in fieldnames(RunMetadata)
        println(io, "# $name\t$(encode_metadata(getfield(metadata, name)))")
    end
    return println(io, join(RESULT_COLUMNS, '\t'))
end

parse_metadata(::Type{String}, value) = value
parse_metadata(::Type{T}, value) where {T<:Union{Int,Float64,Bool}} = parse(T, value)
parse_metadata(::Type{BenchmarkManifest}, value) = decode_manifest(value)

function read_checkpoint(path)
    metadata_values = Dict{Symbol,String}()
    results = Result[]
    found_header = false
    for line in eachline(path)
        isempty(line) && continue
        if startswith(line, "# ")
            key, value = Base.split(line[3:end], '\t'; limit=2)
            key == "posteriordb_format" && begin
                value == CHECKPOINT_FORMAT ||
                    error("unsupported checkpoint format $value in $path")
                continue
            end
            metadata_values[Symbol(key)] = value
        elseif !found_header
            Base.split(line, '\t') == collect(RESULT_COLUMNS) ||
                error("unexpected result columns in $path")
            found_header = true
        else
            fields = Base.split(line, '\t')
            length(fields) == length(RESULT_COLUMNS) ||
                error("malformed result row in $path")
            push!(
                results,
                Result(fields[1], parse(Int, fields[2]), parse.(Float64, fields[3:end])...),
            )
        end
    end
    found_header || error("missing result header in $path")
    missing_fields = setdiff(fieldnames(RunMetadata), keys(metadata_values))
    isempty(missing_fields) ||
        error("missing metadata $(join(missing_fields, ", ")) in $path")
    values = ntuple(fieldcount(RunMetadata)) do index
        name = fieldname(RunMetadata, index)
        return parse_metadata(fieldtype(RunMetadata, index), metadata_values[name])
    end
    return RunMetadata(values...), results
end

function metadata_compatible(left::RunMetadata, right::RunMetadata)
    return all(fieldnames(RunMetadata)) do name
        return name === :shard_index || getfield(left, name) == getfield(right, name)
    end
end

function merge_checkpoints(paths)
    checkpoints = read_checkpoint.(paths)
    isempty(checkpoints) && error("pass at least one checkpoint")
    reference = first(checkpoints)[1]
    all(metadata_compatible(reference, metadata) for (metadata, _) in checkpoints) ||
        error("checkpoint metadata differ")
    indices = sort!([metadata.shard_index for (metadata, _) in checkpoints])
    indices == collect(1:(reference.shard_count)) || error(
        "expected shard indices 1:$(reference.shard_count), received $(join(indices, ", "))",
    )

    results = reduce(vcat, (rows for (_, rows) in checkpoints); init=Result[])
    names = [result.name for result in results]
    length(unique(names)) == length(names) || error("duplicate models in checkpoints")
    Set(names) == Set(POSTERIOR_NAMES) || error(
        "checkpoint coverage differs from POSTERIOR_NAMES: " *
        "missing=$(join(setdiff(POSTERIOR_NAMES, names), ",")); " *
        "unexpected=$(join(setdiff(names, POSTERIOR_NAMES), ","))",
    )
    order = Dict(name => index for (index, name) in pairs(POSTERIOR_NAMES))
    sort!(results; by=result -> order[result.name])
    merged_metadata = RunMetadata(
        (
            name === :shard_index ? 0 : getfield(reference, name) for
            name in fieldnames(RunMetadata)
        )...,
    )
    return results, merged_metadata
end

function stan_version(sm)
    version_match = match(r"(?m)^Stan version:\s*(\S+)", BridgeStan.model_info(sm))
    return version_match === nothing ? "unknown" : only(version_match.captures)
end

function bridge_model(post, seed)
    stan = PosteriorDB.implementation(PosteriorDB.model(post), "stan")
    stan_path = PosteriorDB.path(stan)
    data_json = PosteriorDB.load(PosteriorDB.dataset(post), String)
    stan_library = first(splitext(stan_path)) * "_model.so"
    return BridgeStan.StanModel(
        isfile(stan_library) ? stan_library : stan_path, data_json, seed
    )
end

function random_valid_points(stan_model, rng, draws, scale)
    points = Vector{Vector{Float64}}()
    dimension = Int(BridgeStan.param_unc_num(stan_model))
    attempts = 0
    while length(points) < draws
        attempts += 1
        attempts <= max(100, 100draws) ||
            error("could not draw $draws valid unconstrained parameter vectors")
        q = scale .* randn(rng, dimension)
        valid = try
            isfinite(BridgeStan.log_density(stan_model, q; propto=false, jacobian=false))
        catch
            false
        end
        valid && push!(points, q)
    end
    return points
end

"""
    verify_reactant(posterior_name; draws=30, seed=468)

Check that Wren's Reactant backend reproduces its Enzyme backend's log density
and gradient at random points of the model's own unconstrained space.

Wren and Stan use different unconstrained coordinates, and this benchmark does
not map between them, so Stan is not a reference for these values.
"""
function verify_reactant(posterior_name; draws=30, seed=468)
    draws >= 1 || error("draws must be at least 1")
    post = PosteriorDB.posterior(PDB, posterior_name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    model = make_model(Val(Symbol(posterior_name)), data)

    enzyme = vec_model(model, ENZYME_BACKEND, seed)
    reactant = vec_model(model, REACTANT_BACKEND, seed)

    rng = Xoshiro(seed + sum(codeunits(posterior_name)))
    max_density_error = 0.0
    max_gradient_error = 0.0
    for draw in 1:draws
        q = 0.5 .* randn(rng, enzyme.dimension)
        enzyme_value, enzyme_gradient = Wren.lp_and_grad(enzyme, q)
        reactant_value, reactant_gradient = Wren.lp_and_grad(reactant, q)
        gradient_error = maximum(abs, Array(reactant_gradient) .- Array(enzyme_gradient))
        density_error = abs(Float64(reactant_value) - enzyme_value)
        scale = max(1.0, maximum(abs, Array(enzyme_gradient)))
        density_error <= 1e-8 * max(1.0, abs(enzyme_value)) ||
            error("$posterior_name draw $draw log-density mismatch: $density_error")
        gradient_error <= 1e-7 * scale ||
            error("$posterior_name draw $draw gradient mismatch: $gradient_error")
        max_density_error = max(max_density_error, density_error)
        max_gradient_error = max(max_gradient_error, gradient_error)
    end
    return (; max_density_error, max_gradient_error)
end

function benchmark_one(
    model_name; logdensity_only=false, stan_only=false, wren_only=false, seed=468
)
    post = PosteriorDB.posterior(PDB, model_name)
    stan_model = bridge_model(post, seed)
    dimension = Int(BridgeStan.param_unc_num(stan_model))

    wren = if stan_only
        (primal=NaN, reactant_primal=NaN, enzyme_grad=NaN, reactant_grad=NaN, dim=dimension)
    else
        data = PosteriorDB.load(PosteriorDB.dataset(post))
        model = make_model(Val(Symbol(model_name)), data)
        measured = bench_wren(model, model_name, seed; logdensity_only)
        measured.dim == dimension || error(
            "$model_name: Wren has $(measured.dim) parameters, Stan has $dimension"
        )
        measured
    end

    stan_primal = NaN
    stan_gradient = NaN
    if !wren_only
        q = only(random_valid_points(stan_model, Xoshiro(seed), 1, 0.2))
        BridgeStan.log_density(stan_model, q; propto=false)
        stan_primal = median_time(@be BridgeStan.log_density($stan_model, $q; propto=false))
        if !logdensity_only
            BridgeStan.log_density_gradient(stan_model, q; propto=false)
            stan_gradient = median_time(
                @be BridgeStan.log_density_gradient($stan_model, $q; propto=false)
            )
        end
    end

    return (
        Result(
            model_name,
            wren.dim,
            wren.primal,
            wren.reactant_primal,
            stan_primal,
            wren.enzyme_grad,
            wren.reactant_grad,
            stan_gradient,
        ),
        stan_version(stan_model),
    )
end

function benchmark_models(
    models;
    logdensity_only=false,
    stan_only=false,
    wren_only=false,
    seed=468,
    shard_count=1,
    shard_index=1,
    checkpoint_path="",
)
    isempty(models) && error("no models selected")
    manifest = benchmark_manifest()
    results = Result[]
    metadata = nothing
    for model_name in models
        println(stderr, "Running: $model_name ...")
        result, version = benchmark_one(
            model_name; logdensity_only, stan_only, wren_only, seed
        )
        if metadata === nothing
            metadata = run_metadata(
                version;
                logdensity_only,
                stan_only,
                wren_only,
                shard_count,
                shard_index,
                seed,
                manifest,
            )
            if !isempty(checkpoint_path)
                open(checkpoint_path, "w") do io
                    return write_checkpoint_header(io, metadata)
                end
            end
        elseif version != metadata.stan_version
            error("Stan version changed from $(metadata.stan_version) to $version")
        end
        push!(results, result)
        if !isempty(checkpoint_path)
            open(checkpoint_path, "a") do io
                return write_result(io, result)
            end
        end
    end
    return results, metadata::RunMetadata
end

function verify_models(models; draws=30, seed=468)
    passed = true
    for posterior_name in models
        println(stderr, "Verifying: $posterior_name ...")
        try
            verification = verify_reactant(posterior_name; draws, seed)
            println(
                stderr,
                "  density=$(verification.max_density_error), " *
                "gradient=$(verification.max_gradient_error)",
            )
        catch error
            passed = false
            @error "verification failed" posterior_name exception = (
                error, catch_backtrace()
            )
        end
    end
    return passed
end

# ── Summary and table ──

function time_comparison(results, backend_field, stan_field)
    ratios = Float64[]
    for result in results
        backend_time = getfield(result, backend_field)
        stan_time = getfield(result, stan_field)
        if isfinite(backend_time) &&
            backend_time > 0 &&
            isfinite(stan_time) &&
            stan_time > 0
            push!(ratios, backend_time / stan_time)
        end
    end
    isempty(ratios) && return (successful=0, geometric_mean=NaN, median=NaN, faster=0)
    return (;
        successful=length(ratios),
        geometric_mean=exp(sum(log, ratios) / length(ratios)),
        median=median(ratios),
        faster=count(<(1), ratios),
    )
end

fmt_summary_ratio(ratio) = isfinite(ratio) ? @sprintf("%.2f×", ratio) : "n/a"

function write_summary(io, results; logdensity_only=false, stan_only=false, wren_only=false)
    total = length(results)
    posterior = total == 1 ? "posterior" : "posteriors"
    println(io, "## Summary\n")
    println(io, "- **$total PosteriorDB $posterior** benchmarked.")
    (stan_only || wren_only) && return println(io)

    println(io)
    println(
        io,
        "Ratios are backend time divided by the corresponding Stan time; lower is " *
        "better. Aggregates exclude failed measurements.\n",
    )
    println(
        io,
        "| Workload | Successful measurements | Geometric mean vs Stan | Median vs Stan | Faster than Stan |",
    )
    println(io, "|:--|--:|--:|--:|--:|")
    workloads = [
        ("Wren log density", :wren_primal, :stan_primal),
        ("Wren log density (Reactant)", :wren_reactant_primal, :stan_primal),
    ]
    if !logdensity_only
        push!(workloads, ("Enzyme gradient", :wren_enzyme_grad, :stan_grad))
        push!(workloads, ("Reactant gradient", :wren_reactant_grad, :stan_grad))
    end
    for (name, backend_field, stan_field) in workloads
        comparison = time_comparison(results, backend_field, stan_field)
        println(
            io,
            "| $name | $(comparison.successful) / $total | " *
            "$(fmt_summary_ratio(comparison.geometric_mean)) | " *
            "$(fmt_summary_ratio(comparison.median)) | " *
            "$(comparison.faster) / $(comparison.successful) |",
        )
    end
    return println(io)
end

"""
    write_reference_table(io, results, reference)

The recorded DynamicPPL ratios for the same posteriors, beside the ratios
measured here.
"""
function write_reference_table(io, results, reference)
    covered = [result for result in results if haskey(reference, result.name)]
    isempty(covered) && return nothing
    println(
        io,
        "*Table 2. Each column divides a framework's time by the Stan time for the " *
        "same workload, so lower is better. The Turing and Mooncake columns come " *
        "from `benchmarks/posteriordb.md`, measured on another machine; the ratio " *
        "against Stan is what carries across machines. Turing's models use the " *
        "priors of the Stan program, Wren's substitute several of them.*\n",
    )
    println(
        io,
        "| Model | dim | Wren lp | Turing lp | Enzyme gradient | Reactant gradient | Mooncake gradient |",
    )
    println(io, "|:--|--:|--:|--:|--:|--:|--:|")
    for result in covered
        row = reference[result.name]
        println(
            io,
            "| $(result.name) | $(result.dim) | " *
            "$(fmt_ratio(result.wren_primal, result.stan_primal)) | " *
            "$(fmt_ratio(row.turing_primal, row.stan_primal)) | " *
            "$(fmt_ratio(result.wren_enzyme_grad, result.stan_grad)) | " *
            "$(fmt_ratio(result.wren_reactant_grad, result.stan_grad)) | " *
            "$(fmt_ratio(row.mooncake_grad, row.stan_grad)) |",
        )
    end
    return println(io)
end

function write_table(io, results; logdensity_only=false, stan_only=false, wren_only=false)
    gap = "  "
    eval_headers = if stan_only
        ["Stan"]
    elseif wren_only
        ["Wren", "WrenRct"]
    else
        ["Wren", "WrenRct", "Stan"]
    end
    gradient_headers = if logdensity_only
        String[]
    elseif stan_only
        ["Stan"]
    elseif wren_only
        ["EzyRvs", "RctRvs"]
    else
        ["EzyRvs", "RctRvs", "Stan", "RctRvs / Stan"]
    end
    headers = ["Model", "dim", eval_headers..., gradient_headers...]

    function row_values(result)
        eval_values = if stan_only
            [fmt_time(result.stan_primal)]
        elseif wren_only
            [fmt_time(result.wren_primal), fmt_time(result.wren_reactant_primal)]
        else
            [
                fmt_time(result.wren_primal),
                fmt_time(result.wren_reactant_primal),
                fmt_time(result.stan_primal),
            ]
        end
        gradient_values = if logdensity_only
            String[]
        elseif stan_only
            [fmt_time(result.stan_grad)]
        elseif wren_only
            [fmt_time(result.wren_enzyme_grad), fmt_time(result.wren_reactant_grad)]
        else
            [
                fmt_time(result.wren_enzyme_grad),
                fmt_time(result.wren_reactant_grad),
                fmt_time(result.stan_grad),
                fmt_ratio(result.wren_reactant_grad, result.stan_grad),
            ]
        end
        return [result.name, string(result.dim), eval_values..., gradient_values...]
    end

    rows = [row_values(result) for result in results]
    widths = [
        maximum(length(row[column]) for row in Iterators.flatten(([headers], rows))) for
        column in eachindex(headers)
    ]
    widths[1] = max(widths[1], 16)
    total_w = sum(widths) + length(gap) * (length(widths) - 1)

    function render(row)
        return join(
            (
                column == 1 ? rpad(value, widths[column]) : lpad(value, widths[column]) for
                (column, value) in pairs(row)
            ),
            gap,
        )
    end
    center(value, width) = begin
        left = div(width - length(value), 2)
        lpad(rpad(value, width - left), width)
    end

    eval_columns = length(eval_headers)
    gradient_columns = length(gradient_headers)
    pre_w = widths[1] + length(gap) + widths[2]
    eval_w = sum(widths[3:(2 + eval_columns)]) + length(gap) * (eval_columns - 1)
    gradient_w = if iszero(gradient_columns)
        0
    else
        first_gradient = 3 + eval_columns
        sum(widths[first_gradient:end]) + length(gap) * (gradient_columns - 1)
    end

    println(io)
    println(io, "="^total_w)
    println(
        io,
        rstrip(
            " "^(pre_w + length(gap)) *
            center("eval", eval_w) *
            (iszero(gradient_columns) ? "" : gap * center("gradient", gradient_w)),
        ),
    )
    println(
        io,
        " "^(pre_w + length(gap)) *
        "-"^eval_w *
        (iszero(gradient_columns) ? "" : gap * "-"^gradient_w),
    )
    println(io, render(headers))
    println(io, "-"^total_w)
    for row in rows
        println(io, render(row))
    end
    return println(io, "="^total_w)
end

function write_markdown(path, results, metadata::RunMetadata, reference)
    manifest = metadata.manifest.content
    open(path, "w") do io
        println(io, "# PosteriorDB benchmark results for Wren\n")
        println(
            io,
            "Generated by `julia --project=benchmark_wren benchmark_wren/posteriordb.jl`.\n",
        )
        println(
            io,
            "Stan and Wren are timed at separate reproducible draws, and several " *
            "priors are substituted, so the times are comparable but the log-density " *
            "values are not.\n",
        )
        write_summary(
            io,
            results;
            logdensity_only=metadata.logdensity_only,
            stan_only=metadata.stan_only,
            wren_only=metadata.wren_only,
        )
        println(io, "## Environment\n")
        write_environment(io, metadata)
        println(io)
        println(
            io,
            "*Table 1. Median log-density and gradient evaluation times. Wren: plain " *
            "Julia; WrenRct: Reactant; EzyRvs: Enzyme reverse mode; RctRvs: Reactant " *
            "with Enzyme reverse mode; RctRvs / Stan: RctRvs time divided by Stan " *
            "gradient time (lower is better).*\n",
        )
        println(io, "```text")
        write_table(
            io,
            results;
            logdensity_only=metadata.logdensity_only,
            stan_only=metadata.stan_only,
            wren_only=metadata.wren_only,
        )
        println(io, "```\n")
        if !metadata.stan_only && !metadata.wren_only && !metadata.logdensity_only
            write_reference_table(io, results, reference)
        end
        println(io, "<details>")
        println(io, "<summary>Full benchmark Manifest.toml</summary>\n")
        println(io, "```toml")
        print(io, manifest)
        endswith(manifest, '\n') || println(io)
        println(io, "```")
        return println(io, "</details>")
    end
end

function parse_options(args)
    known = Set((
        "--logdensity-only", # Skip gradients.
        "--stan-only",       # Time only Stan.
        "--wren-only",       # Time only Wren.
        "--merge",           # Merge shard checkpoints.
        "--verify",          # Compare Reactant with Enzyme.
    ))
    options = Set(argument for argument in args if startswith(argument, "--"))
    unknown = setdiff(options, known)
    isempty(unknown) || error("unknown option(s): $(join(sort!(collect(unknown)), ", "))")
    positional = [argument for argument in args if !startswith(argument, "--")]
    merge = "--merge" in options
    verify = "--verify" in options
    merge && verify && error("--merge and --verify are mutually exclusive")
    for option in ("--logdensity-only", "--stan-only", "--wren-only")
        (merge || verify) &&
            option in options &&
            error("$option is only valid when benchmarking")
    end
    "--stan-only" in options &&
        "--wren-only" in options &&
        error("--stan-only and --wren-only are mutually exclusive")
    return (;
        positional,
        merge,
        verify,
        logdensity_only="--logdensity-only" in options,
        stan_only="--stan-only" in options,
        wren_only="--wren-only" in options,
    )
end

function shard_configuration()
    count = parse(Int, get(ENV, "PDB_BENCH_SHARDS", "1"))
    index = parse(Int, get(ENV, "PDB_BENCH_SHARD", "1"))
    1 <= index <= count || error("PDB_BENCH_SHARD must be in 1:PDB_BENCH_SHARDS")
    return count, index
end

function main(args=ARGS)
    options = parse_options(args)
    reference = read_reference(REFERENCE_MARKDOWN)
    if options.merge
        isempty(options.positional) && error("pass checkpoint files after --merge")
        results, metadata = merge_checkpoints(options.positional)
        write_table(
            stdout,
            results;
            logdensity_only=metadata.logdensity_only,
            stan_only=metadata.stan_only,
            wren_only=metadata.wren_only,
        )
        path = get(ENV, "PDB_BENCH_MARKDOWN", joinpath(@__DIR__, "posteriordb.md"))
        write_markdown(path, results, metadata, reference)
        return 0
    end

    shard_count, shard_index = shard_configuration()
    models = if isempty(options.positional)
        POSTERIOR_NAMES[shard_index:shard_count:end]
    else
        options.positional
    end
    println(stderr, "Selected $(length(models)) model(s).")

    if options.verify
        draws = parse(Int, get(ENV, "PDB_TEST_DRAWS", "30"))
        seed = parse(Int, get(ENV, "PDB_TEST_SEED", "468"))
        return verify_models(models; draws, seed) ? 0 : 1
    end

    checkpoint_path = get(ENV, "PDB_BENCH_OUTPUT", "")
    shard_count > 1 &&
        isempty(checkpoint_path) &&
        error("sharded benchmarks require a distinct PDB_BENCH_OUTPUT per shard")
    seed = parse(Int, get(ENV, "PDB_BENCH_SEED", "468"))
    results, metadata = benchmark_models(
        models;
        logdensity_only=options.logdensity_only,
        stan_only=options.stan_only,
        wren_only=options.wren_only,
        seed,
        shard_count,
        shard_index,
        checkpoint_path,
    )
    write_table(
        stdout,
        results;
        logdensity_only=metadata.logdensity_only,
        stan_only=metadata.stan_only,
        wren_only=metadata.wren_only,
    )

    full_catalog = isempty(options.positional) && shard_count == 1
    markdown_path = get(
        ENV, "PDB_BENCH_MARKDOWN", full_catalog ? joinpath(@__DIR__, "posteriordb.md") : ""
    )
    isempty(markdown_path) || write_markdown(markdown_path, results, metadata, reference)
    return 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main())
end
