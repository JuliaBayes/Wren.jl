// Chart.js only includes the parts that are registered, which keeps the bundle smaller.
import {
    CategoryScale,
    Chart,
    LineController,
    LineElement,
    LogarithmicScale,
    PointElement,
    Tooltip,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";

Chart.register(
    CategoryScale,
    LineController,
    LineElement,
    LogarithmicScale,
    PointElement,
    Tooltip,
    zoomPlugin,
);
Chart.defaults.font.family = "Inter, sans-serif";

export { Chart };

/**
 * Marker shape for each series, so that series can be told apart without relying on colour.
 * In the same order as the `--series-N` colours in `app.css`.
 */
export const POINT_STYLES = ["circle", "rect", "triangle", "rectRot", "crossRot", "star"] as const;
export const NUM_SERIES_COLOURS = POINT_STYLES.length;

/**
 * SVG versions of the Chart.js marker shapes, for the legend and the hover info. They are
 * centred on (13, 10).
 */
export const MARKERS: Record<(typeof POINT_STYLES)[number], string> = {
    circle: "M 13 6 a 4 4 0 1 0 0.01 0",
    rect: "M 9.5 6.5 h 7 v 7 h -7 z",
    triangle: "M 13 5.5 L 17.5 13.5 L 8.5 13.5 z",
    rectRot: "M 13 5 L 18 10 L 13 15 L 8 10 z",
    crossRot: "M 10 7 L 16 13 M 16 7 L 10 13",
    star: "M 13 5 V 15 M 8 10 H 18 M 9.5 6.5 L 16.5 13.5 M 16.5 6.5 L 9.5 13.5",
};

/** The value of a CSS variable on the root element (which depends on the theme). */
export function cssVar(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
