import type { Entry } from "./types";

/** Changes smaller than this (in either direction) are treated as noise. */
export const NOISE_THRESHOLD = 0.05;
/** Changes between NOISE_THRESHOLD and this are probably noise, so get only a very faint colour. */
const FAINT_THRESHOLD = 0.1;
/** Colour strength (in %) of the faint band. */
const FAINT_PCT = 15;
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
 * if the change is within the noise threshold, and very faint if it is just above it.
 * Stronger colours mean bigger changes.
 */
export function changeColour(c: Comparison): string | undefined {
    if (c.kind === "broke") return "color-mix(in srgb, var(--slower) 60%, transparent)";
    if (c.kind === "fixed") return "color-mix(in srgb, var(--faster) 60%, transparent)";
    if (c.kind !== "ratio") return undefined;
    // Compare the percentage change (as displayed) to the thresholds, rather than the log
    // ratio, so that e.g. a 9.9% speedup counts as under 10%.
    const change = Math.abs(c.ratio - 1);
    if (change < NOISE_THRESHOLD) return undefined;
    const logRatio = Math.log(c.ratio);
    const colour = logRatio < 0 ? "var(--faster)" : "var(--slower)";
    if (change < FAINT_THRESHOLD) return `color-mix(in srgb, ${colour} ${FAINT_PCT}%, transparent)`;
    // Above the faint band, ramp up from FAINT_PCT to full colour at MAX_FACTOR, so that
    // there is no sudden jump at FAINT_THRESHOLD.
    const lo = Math.log(1 + FAINT_THRESHOLD);
    const strength = Math.min((Math.abs(logRatio) - lo) / (Math.log(MAX_FACTOR) - lo), 1);
    const pct = Math.round(FAINT_PCT + (100 - FAINT_PCT) * Math.max(strength, 0));
    return `color-mix(in srgb, ${colour} ${pct}%, transparent)`;
}
