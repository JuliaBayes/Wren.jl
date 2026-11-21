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

/** The value of a CSS variable on the root element (which depends on the theme). */
export function cssVar(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
