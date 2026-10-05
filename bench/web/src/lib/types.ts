// The format of the JSON files written by `bench/suite/run.jl`.

export interface RunMeta {
    timestamp: string;
    git_sha: string | null;
    julia_version: string;
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
}

/** One line of `runs/index.json`. */
export interface IndexEntry {
    file: string;
    timestamp: string;
    git_sha: string | null;
}
