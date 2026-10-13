// Anything beyond 3 significant figures is just noise.
const SIGFIGS = 3;

function sigfigs(x: number): string {
    // Number(...) drops trailing zeros, so that e.g. 120.0 is shown as 120.
    return String(Number(x.toPrecision(SIGFIGS)));
}

/** Format a time in seconds, e.g. `1.23 μs`. */
export function formatTime(seconds: number): string {
    // Round before picking the unit, so that e.g. 999.7 ns is shown as 1 μs, not 1000 ns.
    const x = Number(seconds.toPrecision(SIGFIGS));
    if (x < 1e-6) return `${sigfigs(x * 1e9)} ns`;
    if (x < 1e-3) return `${sigfigs(x * 1e6)} μs`;
    if (x < 1) return `${sigfigs(x * 1e3)} ms`;
    return `${sigfigs(x)} s`;
}

/** Format a ratio tip/base as a percentage change, e.g. `+12.3%`. */
export function formatChange(ratio: number): string {
    const pct = (ratio - 1) * 100;
    const sign = pct > 0 ? "+" : pct < 0 ? "−" : "±";
    return `${sign}${Math.abs(pct).toFixed(1)}%`;
}

export function formatTimestamp(ts: string): string {
    const d = new Date(ts);
    return isNaN(d.getTime()) ? ts : d.toLocaleString();
}

export function shortSha(sha: string | null): string {
    return sha ? sha.slice(0, 7) : "unknown";
}

/** The label of a run, falling back to the short sha for files that don't have one. */
export function runLabel(meta: { label?: string; git_sha: string | null }): string {
    return meta.label ?? shortSha(meta.git_sha);
}

/** Whether a run is of a release, i.e. labelled e.g. "WrenCore-v0.2.0". */
export function isRelease(meta: { label?: string }): boolean {
    return meta.label?.startsWith("WrenCore") ?? false;
}
