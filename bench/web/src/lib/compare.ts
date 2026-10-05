import type { Entry } from "./types";

/** Changes smaller than this (in either direction) are treated as noise. */
export const NOISE_THRESHOLD = 0.05;
/** Changes of this factor or more get the strongest colour. */
const MAX_FACTOR = 2;

export type Comparison =
    | { kind: "no-base" } // nothing to compare against
    | { kind: "both-error" }
    | { kind: "broke"; baseTime: number } // worked in base, errors in tip
    | { kind: "fixed"; baseError: string } // errored in base, works in tip
    | { kind: "ratio"; baseTime: number; ratio: number };

export function compare(tip: Entry | undefined, base: Entry | undefined): Comparison {
    if (tip === undefined || base === undefined) return { kind: "no-base" };
    if ("error" in tip) {
        return "error" in base
            ? { kind: "both-error" }
            : { kind: "broke", baseTime: base.time };
    }
    if ("error" in base) return { kind: "fixed", baseError: base.error };
    return { kind: "ratio", baseTime: base.time, ratio: tip.time / base.time };
}

/**
 * Background colour for a cell: green if tip is faster than base, red if slower, nothing
 * if the change is within the noise threshold. Stronger colours mean bigger changes.
 */
export function changeColour(c: Comparison): string | undefined {
    if (c.kind === "broke") return "color-mix(in srgb, var(--slower) 60%, transparent)";
    if (c.kind === "fixed") return "color-mix(in srgb, var(--faster) 60%, transparent)";
    if (c.kind !== "ratio") return undefined;
    const logRatio = Math.log(c.ratio);
    if (Math.abs(logRatio) < Math.log(1 + NOISE_THRESHOLD)) return undefined;
    const strength = Math.min(Math.abs(logRatio) / Math.log(MAX_FACTOR), 1);
    const pct = Math.round(25 + 75 * strength);
    const colour = logRatio < 0 ? "var(--faster)" : "var(--slower)";
    return `color-mix(in srgb, ${colour} ${pct}%, transparent)`;
}
