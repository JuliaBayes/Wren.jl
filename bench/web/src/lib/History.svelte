<script lang="ts">
    import { fetchRun } from "./load";
    import { POINT_STYLES, NUM_SERIES_COLOURS } from "./chart";
    import { modelUrl, runSha } from "./github";
    import { allBackends, modelHistories, sortRuns, type Range } from "./history";
    import HistoryChart from "./HistoryChart.svelte";
    import type { IndexEntry, Run } from "./types";

    interface Props {
        index: IndexEntry[];
        theme: string;
        /** Called with the files of the runs to compare (base is "" for none). */
        onCompare: (tip: string, base: string) => void;
    }
    let { index, theme, onCompare }: Props = $props();

    let runs: Run[] = $state([]);
    let loading = $state(true);
    let failed: string[] = $state([]);

    $effect(() => {
        const entries = index;
        if (entries.length === 0) return;
        loading = true;
        Promise.allSettled(entries.map((e) => fetchRun(e.file))).then((results) => {
            if (entries !== index) return;
            runs = sortRuns(results.flatMap((r) => (r.status === "fulfilled" ? [r.value] : [])));
            failed = entries.filter((_, i) => results[i].status === "rejected").map((e) => e.file);
            loading = false;
        });
    });

    const histories = $derived(modelHistories(runs));
    // Each backend keeps the same colour in every chart.
    const backends = $derived(allBackends(runs));
    const slots = $derived(new Map(backends.map((b, i) => [b, i % NUM_SERIES_COLOURS])));
    const newestSha = $derived(runs.length > 0 ? runSha(runs[runs.length - 1].meta) : null);

    // The zoomed range, shared by all charts. Reset when the set of runs changes, since the
    // indices then refer to different runs.
    let range: Range = $state(null);
    $effect(() => {
        void runs;
        range = null;
    });

    // Backends hidden by clicking them in the legend, in all charts.
    let hidden: Set<string> = $state(new Set());
    function toggle(backend: string) {
        const next = new Set(hidden);
        if (!next.delete(backend)) next.add(backend);
        hidden = next;
    }

    function pick(i: number) {
        onCompare(runs[i].source, i > 0 ? runs[i - 1].source : "");
    }

    // SVG versions of the Chart.js marker shapes, for the legend.
    const MARKERS: Record<(typeof POINT_STYLES)[number], string> = {
        circle: "M 13 6 a 4 4 0 1 0 0.01 0",
        rect: "M 9.5 6.5 h 7 v 7 h -7 z",
        triangle: "M 13 5.5 L 17.5 13.5 L 8.5 13.5 z",
        rectRot: "M 13 5 L 18 10 L 13 15 L 8 10 z",
        crossRot: "M 10 7 L 16 13 M 16 7 L 10 13",
        star: "M 13 5 V 15 M 8 10 H 18 M 9.5 6.5 L 16.5 13.5 M 16.5 6.5 L 9.5 13.5",
    };
</script>

{#if loading && runs.length === 0}
    <p>Loading runs…</p>
{:else if runs.length === 0}
    <p>No runs found.</p>
{:else}
    <ul class="legend">
        <li>
            Each chart shows one model, with the fastest time of each run (smaller is faster).
            Runs are in commit order; dashed lines mark releases.
        </li>
        <li>
            Drag to pan, and Ctrl + scroll (or pinch) to zoom.
        </li>
    </ul>

    <div class="controls">
        <div class="keys">
            <span class="keys-title">Show/hide:</span>
            {#each backends as backend}
                {@const slot = slots.get(backend) ?? 0}
                <label class="key" class:off={hidden.has(backend)}>
                    <input type="checkbox" checked={!hidden.has(backend)} onchange={() => toggle(backend)} />
                    <svg width="26" height="20" viewBox="0 0 26 20" aria-hidden="true">
                        <line x1="1" y1="10" x2="25" y2="10" style:stroke={`var(--series-${slot + 1})`} stroke-width="2" />
                        <path
                            d={MARKERS[POINT_STYLES[slot]]}
                            style:fill={`var(--series-${slot + 1})`}
                            style:stroke={`var(--series-${slot + 1})`}
                            stroke-width="1.5"
                        />
                    </svg>
                    {backend}
                </label>
            {/each}
        </div>
        <button class="reset" onclick={() => (range = null)} disabled={range === null}>Reset zoom</button>
    </div>

    {#if failed.length > 0}
        <p class="error">Could not load: {failed.join(", ")}</p>
    {/if}

    <div class="grid">
        {#each histories as h (h.model)}
            {@const url = modelUrl(h.model, newestSha)}
            <section>
                <h3>
                    {#if url}
                        <a href={url} target="_blank" rel="noreferrer">{h.model}</a>
                    {:else}
                        {h.model}
                    {/if}
                    {#if h.dim !== null}<span class="dim">dim {h.dim}</span>{/if}
                </h3>
                <HistoryChart
                    {runs}
                    series={h.series}
                    {slots}
                    {hidden}
                    {theme}
                    {range}
                    onRangeChange={(r) => (range = r)}
                    onPick={pick}
                />
            </section>
        {/each}
    </div>
{/if}

<style>
    .legend {
        font-size: 0.9rem;
        padding-left: 18px;
        color: var(--text-secondary);
    }
    .controls {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px 16px;
        margin-bottom: 8px;
    }
    .keys {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 8px;
        font-family: "Fira Code", monospace;
        font-size: 0.85rem;
    }
    .keys-title {
        font-family: "Inter", sans-serif;
        font-size: 0.9rem;
        color: var(--text-secondary);
        align-self: center;
    }
    .key {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 10px;
        border: 1px solid var(--table-border);
        border-radius: 6px;
        cursor: pointer;
    }
    .key:hover {
        background-color: var(--btn-hover);
    }
    .key input {
        margin: 0;
        cursor: pointer;
    }
    /* Fade the swatch and name of hidden backends, but not the checkbox. */
    .key.off svg {
        opacity: 0.35;
    }
    .key.off {
        color: var(--text-secondary);
    }
    .reset {
        margin-left: auto;
        font: inherit;
        font-size: 0.9rem;
        padding: 4px 8px;
        border: 1px solid var(--table-border);
        border-radius: 6px;
        background-color: var(--btn-bg);
        color: var(--btn-text);
        cursor: pointer;
    }
    .reset:hover:not(:disabled) {
        background-color: var(--btn-hover);
    }
    .reset:disabled {
        opacity: 0.5;
        cursor: default;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(min(340px, 100%), 1fr));
        gap: 16px;
    }
    section {
        border: 1px solid var(--table-border);
        border-radius: 8px;
        padding: 8px 12px 12px;
        /* Lets the chart shrink with its column. */
        min-width: 0;
    }
    h3 {
        margin: 0 0 4px;
        font-size: 1rem;
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .dim {
        font-size: 0.8rem;
        font-weight: normal;
        color: var(--text-secondary);
    }
    .error {
        color: var(--error-color);
    }
</style>
