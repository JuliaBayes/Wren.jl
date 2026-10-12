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

// The file defining each model in the benchmark results. The names here must match the
// names in `MODELS` in `bench/suite/run.jl`.
const MODEL_DIR = "WrenCore/ext/WrenCorePosteriorDBExt/models";
const MODEL_FILES = new Map<string, string>([
    ["ArK", `${MODEL_DIR}/ark.jl`],
    ["EightSchoolsCentered", `${MODEL_DIR}/eight_schools_centered.jl`],
    ["GLMBinomial", `${MODEL_DIR}/glm_binomial.jl`],
    ["GLMPoisson", `${MODEL_DIR}/glm_poisson.jl`],
    ["GLMMPoisson", `${MODEL_DIR}/glmm_poisson.jl`],
    ["GPRegr", `${MODEL_DIR}/gp_regr.jl`],
    ["KidscoreInteraction", `${MODEL_DIR}/kidscore_interaction.jl`],
    ["Mh", `${MODEL_DIR}/mh.jl`],
    ["Rate1", `${MODEL_DIR}/rate1.jl`],
    ["SblriBLR", `${MODEL_DIR}/sblri_blr.jl`],
]);

/**
 * Link to the definition of a model at the given commit (or on main if the commit is
 * unknown), or null if the model's file can't be found.
 */
export function modelUrl(name: string, sha: string | null): string | null {
    const file = MODEL_FILES.get(name);
    return file === undefined ? null : `${REPO_URL}/blob/${sha ?? "main"}/${file}`;
}
