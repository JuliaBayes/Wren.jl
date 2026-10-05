// The format of the JSON files written by `bench/suite/run.jl`.

export interface RunMeta {
    /** When the benchmark was run. */
    timestamp: string;
    git_sha: string | null;
    /** When the benchmarked commit was made. Missing in older files. */
    commit_date?: string | null;
    /** Identifies the run, e.g. "WrenCore-v0.2.0" or "PR #123". Missing in older files. */
    label?: string;
    /** Whether there were uncommitted changes. Missing in older files. */
    dirty?: boolean;
    julia_version: string;
    /** The machine the benchmarks were run on. Missing in older files. */
    machine?: {
        cpu: string;
        cpu_threads: number;
        memory_gb: number;
        os: string;
        arch: string;
        julia_threads?: number;
        blas_threads?: number;
    };
}

/** The result of benchmarking one model with one backend. Times are in seconds. */
export type Entry = { time: number } | { error: string };

export interface ModelResult {
    dim: number | null;
    entries: Map<string, Entry>;
}

export interface Run {
    /** Where the run was loaded from (a file name). */
    source: string;
    meta: RunMeta;
    /** Backend names, in the order they appear in the file. */
    backends: string[];
    models: Map<string, ModelResult>;
    /** Package name => version, or null if the file has no manifest. */
    manifest: Map<string, string | null> | null;
}

/** One line of `runs/index.json`. */
export interface IndexEntry {
    file: string;
    timestamp: string;
    git_sha: string | null;
    commit_date?: string | null;
    label?: string;
}
