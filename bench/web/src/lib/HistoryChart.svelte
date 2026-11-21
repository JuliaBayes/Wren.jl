<script lang="ts">
    import { untrack } from "svelte";
    import type { Plugin } from "chart.js";
    import { Chart, POINT_STYLES, cssVar } from "./chart";
    import { formatTime, formatTimestamp, isRelease, runLabel, shortSha } from "./format";
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
        /** Called with the index of a run when it is clicked. */
        onPick: (i: number) => void;
    }
    let { runs, series, slots, hidden, theme, range, onRangeChange, onPick }: Props = $props();

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

        // Dashed vertical lines marking releases.
        const releaseLines: Plugin<"line"> = {
            id: "releaseLines",
            beforeDatasetsDraw(c) {
                const { ctx, chartArea, scales } = c;
                ctx.save();
                ctx.strokeStyle = text;
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                for (const i of releases) {
                    const x = scales.x.getPixelForValue(i);
                    if (x < chartArea.left || x > chartArea.right) continue;
                    ctx.beginPath();
                    ctx.moveTo(x, chartArea.top);
                    ctx.lineTo(x, chartArea.bottom);
                    ctx.stroke();
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
            plugins: [releaseLines],
            options: {
                animation: false,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                scales: {
                    x: {
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
                    tooltip: {
                        callbacks: {
                            title: (items) => {
                                const r = runs[items[0].dataIndex];
                                // The tooltip is drawn inside the chart, so long labels
                                // would be cut off.
                                const label = runLabel(r.meta);
                                return [
                                    label.length > 45 ? `${label.slice(0, 44)}…` : label,
                                    `${shortSha(r.meta.git_sha)} · ${formatTimestamp(r.meta.commit_date ?? r.meta.timestamp)} · Julia ${r.meta.julia_version}`,
                                ];
                            },
                            label: (item) => ` ${item.dataset.label}: ${formatTime(item.parsed.y ?? NaN)}`,
                            footer: (items) => {
                                const i = items[0].dataIndex;
                                const errored = series.filter((s) => !hidden.has(s.backend) && s.errors[i] !== null);
                                return [
                                    ...errored.map((s) => `${s.backend}: error`),
                                    i > 0 ? "Click to compare with the previous run" : "Click to view this run",
                                ];
                            },
                        },
                    },
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
                onClick: (e) => {
                    if (Date.now() - lastPanEnd < 300 || e.x === null) return;
                    const i = Math.round(c.scales.x.getValueForPixel(e.x) ?? -1);
                    if (i >= 0 && i < runs.length) onPick(i);
                },
            },
        });
        chart = c;
        untrack(() => applyRange(range));
        return () => {
            c.destroy();
            chart = null;
        };
    });

    // Follow zooming and panning in the other charts.
    $effect(() => applyRange(range));
</script>

<div class="chart">
    <canvas bind:this={canvas}></canvas>
</div>

<style>
    .chart {
        position: relative;
        height: 220px;
    }
    canvas {
        cursor: pointer;
    }
</style>
