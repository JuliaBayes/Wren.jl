import type { Entry, IndexEntry, ModelResult, Run } from "./types";

const SUPPORTED_SCHEMA_VERSION = 1;

/** Turn the contents of a results JSON file into a `Run`. */
export function parseRun(source: string, json: any): Run {
    if (json.schema_version !== SUPPORTED_SCHEMA_VERSION) {
        throw new Error(
            `${source}: unsupported schema_version ${json.schema_version}`,
        );
    }
    // Older or hand-written files might not have a `backends` list; in that case the
    // backends are collected from the results below.
    const backends: string[] = [...(json.backends ?? [])];
    const models = new Map<string, ModelResult>();
    for (const [name, raw] of Object.entries<any>(json.results)) {
        const entries = new Map<string, Entry>();
        for (const [key, value] of Object.entries<any>(raw)) {
            if (key === "dim") continue;
            entries.set(key, value as Entry);
            if (!backends.includes(key)) backends.push(key);
        }
        models.set(name, { dim: raw.dim ?? null, entries });
    }
    const manifest = json.manifest
        ? new Map(Object.entries<string | null>(json.manifest))
        : null;
    return { source, meta: json.meta, backends, models, manifest };
}

/** Fetch `runs/index.json`. Returns an empty list if there isn't one. */
export async function fetchIndex(): Promise<IndexEntry[]> {
    try {
        const res = await fetch("runs/index.json");
        if (!res.ok) return [];
        return await res.json();
    } catch {
        return [];
    }
}

// Saved runs don't change, so each one only needs to be fetched once (the history view
// fetches all of them, and the compare view then reuses them).
const runCache = new Map<string, Promise<Run>>();

export function fetchRun(file: string): Promise<Run> {
    let run = runCache.get(file);
    if (run === undefined) {
        run = fetchRunUncached(file);
        // Don't keep failures, so that they can be retried.
        run.catch(() => runCache.delete(file));
        runCache.set(file, run);
    }
    return run;
}

async function fetchRunUncached(file: string): Promise<Run> {
    const res = await fetch(`runs/${file}`);
    if (!res.ok) throw new Error(`Could not fetch runs/${file} (${res.status})`);
    return parseRun(file, await res.json());
}

export async function readRunFile(file: File): Promise<Run> {
    return parseRun(file.name, JSON.parse(await file.text()));
}
