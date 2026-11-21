<script lang="ts">
    import { untrack } from "svelte";
    import type { Plugin } from "chart.js";
    import { Chart, MARKERS, POINT_STYLES, cssVar } from "./chart";
    import { formatTime, isRelease, runLabel, shortSha } from "./format";
    import type { Range, Series } from "./history";
    import type { Run } from "./types";

    interface Props {
        /** The runs, oldest first. */
        runs: Run[];
        series: Series[];
        /** The colour slot (index into `--series-N`) of each backend. */
        slots: Map<string, number>;
        /** Backends whose lines are hidden. The y-axis fits only the lines that are shown. */
        hidden: Set<string>;
        /** Changes whenever the theme does, so that the chart is redrawn in the new colours. */
        theme: string;
        /** The range of run indices shown, or null for all of them. Shared by all charts. */
        range: Range;
        onRangeChange: (range: Range) => void;
        /** The index of the run under the mouse (in any chart), or null. Shared by all charts. */
        hovered: number | null;
        onHover: (i: number | null) => void;
        /** The index of the run selected by clicking (in any chart), or null. Shared by all charts. */
        selected: number | null;
        /** Called with the index of a run when it is clicked. */
        onSelect: (i: number) => void;
    }
    let {
        runs,
        series,
        slots,
        hidden,
        theme,
        range,
        onRangeChange,
        hovered,
        onHover,
        selected,
        onSelect,
    }: Props = $props();

    let canvas: HTMLCanvasElement;
    let chart: Chart<"line", (number | null)[], string> | null = null;
    // Panning with the mouse ends with a click, which shouldn't count as picking a run.
    let lastPanEnd = 0;

    /**
     * When the y-axis covers less than this many powers of ten, Chart.js's log ticks are
     * mostly bunched up at one end, so evenly spaced round numbers are used instead.
     */
    const NARROW_DECADES = 0.5;

    /** About `count` evenly spaced round numbers between `min` and `max`. */
    function niceTicks(min: number, max: number, count = 5): number[] {
        const raw = (max - min) / count;
        const p = 10 ** Math.floor(Math.log10(raw));
        const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw)!;
        const ticks = [];
        for (let i = Math.ceil(min / step); i * step <= max; i++) {
            // Rounding avoids floating-point noise such as 0.30000000000000004.
            ticks.push(Number((i * step).toPrecision(12)));
        }
        return ticks;
    }

    /**
     * Label for a tick on the log axis. Chart.js puts a tick at every 1, 2, ..., 9 times a
     * power of ten, which is too many to label, so only some are labelled: fewer when the
     * axis covers a wider range, so that the labels don't overlap. (Ticks from `niceTicks`
     * are all labelled.)
     */
    function logTick(value: number | string, _i: number, ticks: { value: number }[]): string {
        const v = Number(value);
        const values = ticks.map((t) => t.value);
        const decades = Math.log10(Math.max(...values) / Math.min(...values));
        if (decades < NARROW_DECADES) return formatTime(v);
        const shown = decades > 1.2 ? [1, 2, 5] : decades > 0.5 ? [1, 2, 3, 5] : [1, 2, 3, 4, 5, 6, 7, 8, 9];
        const mantissa = v / 10 ** Math.floor(Math.log10(v));
        return shown.some((m) => Math.abs(mantissa - m) < 1e-6) ? formatTime(v) : "";
    }

    /**
     * The index of the shown run nearest to `x` (in pixels), or null if `x` is over the y-axis.
     * Anything right of the last run counts as the last run, so that it's easy to hover.
     */
    function runAt(x: number): number | null {
        if (chart === null || x < chart.chartArea.left) return null;
        const { min, max } = chart.scales.x;
        const i = Math.round(chart.scales.x.getValueForPixel(x) ?? -1);
        return Math.min(Math.max(i, min), max);
    }

    // Copies of `hovered` and `selected` for the plugin that draws lines through them, since
    // plugins can't depend on Svelte state.
    let hoverLine: number | null = null;
    let selectedLine: number | null = null;

    /** Highlight the points of the hovered (or else the selected) run. */
    function showHover() {
        if (chart === null) return;
        const c = chart;
        const i = hoverLine ?? selectedLine;
        const active =
            i === null
                ? []
                : c.data.datasets.flatMap((d, k) =>
                      c.isDatasetVisible(k) && d.data[i] != null ? [{ datasetIndex: k, index: i }] : [],
                  );
        c.setActiveElements(active);
        c.update("none");
    }

    $effect(() => {
        hoverLine = hovered;
        selectedLine = selected;
        showHover();
    });

    // The run whose times are shown below the chart: the hovered one, or else the selected
    // one. Nothing is shown if there is neither.
    const shown = $derived(hovered ?? selected);

    function reportRange() {
        if (chart === null) return;
        const { min, max } = chart.scales.x;
        onRangeChange(min === 0 && max === runs.length - 1 ? null : { min, max });
    }

    function applyRange(r: Range) {
        if (chart === null) return;
        const { min, max } = chart.scales.x;
        if (r === null) {
            if (min !== 0 || max !== runs.length - 1) chart.resetZoom("none");
        } else if (r.min !== min || r.max !== max) {
            chart.zoomScale("x", r, "none");
        }
    }

    // Canvas text is drawn with whatever fonts have loaded so far, and Chart.js remembers the
    // sizes of labels, so the chart is made again once all fonts have loaded.
    let fontsLoaded = $state(false);
    document.fonts.ready.then(() => (fontsLoaded = true));

    // (Re)create the chart whenever the data, the hidden backends, the theme, or the fonts
    // change.
    $effect(() => {
        void theme;
        void fontsLoaded;
        const text = cssVar("--text-secondary");
        const border = cssVar("--table-border");
        const releases = runs.flatMap((r, i) => (isRelease(r.meta) ? [i] : []));

        // Dashed vertical lines marking releases, and solid ones through the selected and
        // hovered runs.
        const verticalLines: Plugin<"line"> = {
            id: "verticalLines",
            beforeDatasetsDraw(c) {
                const { ctx, chartArea, scales } = c;
                const line = (i: number) => {
                    const x = scales.x.getPixelForValue(i);
                    if (x < chartArea.left || x > chartArea.right) return;
                    ctx.beginPath();
                    ctx.moveTo(x, chartArea.top);
                    ctx.lineTo(x, chartArea.bottom);
                    ctx.stroke();
                };
                ctx.save();
                ctx.strokeStyle = text;
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                releases.forEach(line);
                ctx.setLineDash([]);
                if (selectedLine !== null) {
                    ctx.lineWidth = 2;
                    line(selectedLine);
                }
                if (hoverLine !== null && hoverLine !== selectedLine) {
                    ctx.lineWidth = 1;
                    ctx.globalAlpha = 0.5;
                    line(hoverLine);
                }
                ctx.restore();
            },
        };

        const c = new Chart(canvas, {
            type: "line",
            data: {
                // Releases are labelled with their version, other runs with their sha.
                labels: runs.map((r) =>
                    isRelease(r.meta) ? runLabel(r.meta).replace(/^WrenCore-/, "") : shortSha(r.meta.git_sha),
                ),
                datasets: series.map((s) => {
                    const slot = slots.get(s.backend) ?? 0;
                    const colour = cssVar(`--series-${slot + 1}`);
                    return {
                        label: s.backend,
                        data: s.times,
                        hidden: hidden.has(s.backend),
                        borderColor: colour,
                        backgroundColor: colour,
                        borderWidth: 2,
                        pointStyle: POINT_STYLES[slot],
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        // Runs where this backend errored are left as a gap.
                        spanGaps: false,
                    };
                }),
            },
            plugins: [verticalLines],
            options: {
                animation: false,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                scales: {
                    x: {
                        // Leave half a run's width at each end, so that the first and last
                        // runs aren't on the edge of the chart and are easy to hover and click.
                        offset: true,
                        ticks: { color: text, maxRotation: 0, font: { family: "Fira Code", size: 10 } },
                        grid: { display: false },
                        border: { color: border },
                    },
                    y: {
                        type: "logarithmic",
                        // `logTick` already leaves out labels that would overlap.
                        ticks: { color: text, callback: logTick, autoSkip: false },
                        afterBuildTicks: (scale) => {
                            if (Math.log10(scale.max / scale.min) >= NARROW_DECADES) return;
                            const ticks = niceTicks(scale.min, scale.max);
                            if (ticks.length >= 2) scale.ticks = ticks.map((value) => ({ value }));
                        },
                        grid: { color: border },
                        border: { display: false },
                    },
                },
                plugins: {
                    // Hover info is shown below the chart instead, so that it doesn't
                    // cover the lines.
                    tooltip: { enabled: false },
                    zoom: {
                        // Only the x-axis is zoomed; the y-axis fits the runs that are shown.
                        pan: {
                            enabled: true,
                            mode: "x",
                            onPanComplete: () => {
                                lastPanEnd = Date.now();
                                reportRange();
                            },
                        },
                        zoom: {
                            // Plain scrolling scrolls the page. (Pinching on a trackpad
                            // counts as scrolling with Ctrl held.)
                            wheel: { enabled: true, modifierKey: "ctrl" },
                            pinch: { enabled: true },
                            mode: "x",
                            onZoomComplete: reportRange,
                        },
                        limits: { x: { min: "original", max: "original", minRange: 1 } },
                    },
                },
            },
        });
        chart = c;
        untrack(() => applyRange(range));
        showHover();
        return () => {
            c.destroy();
            chart = null;
        };
    });

    // Follow zooming and panning in the other charts.
    $effect(() => applyRange(range));
</script>

<div class="chart">
    <canvas
        bind:this={canvas}
        onmousemove={(e) => onHover(runAt(e.offsetX))}
        onmouseleave={() => onHover(null)}
        onclick={(e) => {
            // Chart.js ignores clicks outside the area where the lines are drawn, so clicks
            // are handled here instead.
            if (Date.now() - lastPanEnd < 300) return;
            const i = runAt(e.offsetX);
            if (i !== null) onSelect(i);
        }}
    ></canvas>
</div>

<!-- This model's times for the hovered (or else selected) run. Always present (even if
empty), so that the chart doesn't move. -->
<div class="times">
    {#if shown !== null}
        {#each series.filter((s) => !hidden.has(s.backend)) as s (s.backend)}
            {@const slot = slots.get(s.backend) ?? 0}
            {@const time = s.times[shown]}
            <span class="time" title={s.backend}>
                <svg width="12" height="12" viewBox="7 4 12 12" aria-hidden="true">
                    <path
                        d={MARKERS[POINT_STYLES[slot]]}
                        style:fill={`var(--series-${slot + 1})`}
                        style:stroke={`var(--series-${slot + 1})`}
                        stroke-width="1.5"
                    />
                </svg>
                {#if time !== null}
                    {formatTime(time)}
                {:else if s.errors[shown] !== null}
                    <span class="error">error</span>
                {:else}
                    –
                {/if}
            </span>
        {/each}
    {/if}
</div>

<style>
    .chart {
        position: relative;
        height: 220px;
    }
    canvas {
        cursor: pointer;
    }
    .times {
        margin-top: 6px;
        min-height: 1.6em;
        font-size: 0.8rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0 14px;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }
    .time {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .error {
        color: var(--error-color);
    }
</style>
