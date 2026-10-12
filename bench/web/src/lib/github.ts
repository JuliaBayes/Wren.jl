import type { RunMeta } from "./types";

const REPO_URL = "https://github.com/JuliaBayes/Wren.jl";

/**
 * The commit that a run's code is at, or null if it's unknown. Runs with uncommitted
 * changes count as unknown, since no commit has their code. (These can only be local
 * files, because `run.jl` refuses to save them.)
 */
export function runSha(meta: RunMeta): string | null {
    return meta.dirty ? null : meta.git_sha;
}

export function commitUrl(sha: string): string {
    return `${REPO_URL}/commit/${sha}`;
}

export function compareUrl(baseSha: string, tipSha: string): string {
    return `${REPO_URL}/compare/${baseSha}...${tipSha}`;
}

// Models are identified in the results by their module name (e.g. `ArK`), not their
// file name, so read the model files at build time to match them up.
const MODEL_SOURCES = import.meta.glob<string>("../../../../WrenCore/ext/WrenCorePosteriorDBExt/models/*.jl", {
    query: "?raw",
    import: "default",
    eager: true,
});
const MODEL_FILES = new Map<string, string>();
for (const [path, source] of Object.entries(MODEL_SOURCES)) {
    const name = source.match(/^module\s+(\w+)/m)?.[1];
    if (name !== undefined) {
        MODEL_FILES.set(name, `WrenCore/ext/WrenCorePosteriorDBExt/models/${path.split("/").pop()}`);
    }
}

/**
 * Link to the definition of a model at the given commit (or on main if the commit is
 * unknown), or null if the model's file can't be found.
 */
export function modelUrl(name: string, sha: string | null): string | null {
    const file = MODEL_FILES.get(name);
    return file === undefined ? null : `${REPO_URL}/blob/${sha ?? "main"}/${file}`;
}
