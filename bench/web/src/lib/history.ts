import type { Run } from "./types";

/** The times of one backend on one model, with one element per run (oldest first). */
export interface Series {
    backend: string;
    /** null if the run has no time for this backend (it errored, or wasn't run). */
    times: (number | null)[];
    errors: (string | null)[];
}

/** A range of run indices, as shown on a chart's x-axis, or null for all runs. */
export type Range = { min: number; max: number } | null;

export interface ModelHistory {
    model: string;
    dim: number | null;
    series: Series[];
}

/**
 * Describes the machine a run was on, or null if it's unknown (older files). Runs on
 * different machines can't be compared.
 */
export function machineKey(run: Run): string | null {
    const m = run.meta.machine;
    return m ? `${m.cpu}, ${m.memory_gb} GB, ${m.os} ${m.arch}` : null;
}

/**
 * Sort runs oldest first, by the date of the benchmarked commit (falling back to when the
 * benchmark was run, for older files). This keeps runs in the order of the git history even
 * if older commits were benchmarked later.
 */
export function sortRuns(runs: Run[]): Run[] {
    const key = (r: Run) => r.meta.commit_date ?? r.meta.timestamp;
    return [...runs].sort(
        (a, b) =>
            key(a).localeCompare(key(b)) || a.meta.timestamp.localeCompare(b.meta.timestamp),
    );
}

/**
 * Collect names in the order they appear in the newest run, followed by any that only appear
 * in older runs.
 */
function collect(runs: Run[], names: (r: Run) => Iterable<string>): string[] {
    const out = new Set<string>();
    for (let i = runs.length - 1; i >= 0; i--) for (const n of names(runs[i])) out.add(n);
    return [...out];
}

/** All backends in `runs` (which must be sorted oldest first). */
export function allBackends(runs: Run[]): string[] {
    return collect(runs, (r) => r.backends);
}

/** The history of every model in `runs` (which must be sorted oldest first). */
export function modelHistories(runs: Run[]): ModelHistory[] {
    const backends = allBackends(runs);
    return collect(runs, (r) => r.models.keys()).map((model) => {
        const results = runs.map((r) => r.models.get(model));
        const series = backends
            .map((backend): Series => {
                const entries = results.map((m) => m?.entries.get(backend));
                return {
                    backend,
                    times: entries.map((e) => (e && "time" in e ? e.time : null)),
                    errors: entries.map((e) => (e && "error" in e ? e.error : null)),
                };
            })
            // Leave out backends that were never run on this model.
            .filter((s) => s.times.some((t) => t !== null) || s.errors.some((e) => e !== null));
        const dim = results.findLast((m) => m !== undefined)?.dim ?? null;
        return { model, dim, series };
    });
}
