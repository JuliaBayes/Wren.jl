<script lang="ts">
    import { changeColour, compare, type Comparison } from "./compare";
    import { formatChange, formatTime } from "./format";
    import { modelUrl } from "./github";
    import type { Entry, Run } from "./types";

    interface Props {
        tip: Run;
        base: Run | null;
    }
    let { tip, base }: Props = $props();

    // The backend that AD timings are divided by to get the "× primal" ratio.
    const PRIMAL = "primal";

    // Sorting: by model name, by dimension, or by the tip time of a backend. Backends
    // are identified by `backend:<name>`.
    type SortColumn = "name" | "dim" | `backend:${string}`;
    let sortColumn: SortColumn = $state("name");
    let sortAscending = $state(true);

    function setSort(column: SortColumn) {
        if (sortColumn === column) {
            sortAscending = !sortAscending;
        } else {
            sortColumn = column;
            sortAscending = true;
        }
    }

    function timeOf(entry: Entry | undefined): number {
        return entry !== undefined && "time" in entry ? entry.time : Infinity;
    }

    // The value that a model is sorted by. Missing values (errors, no dim) are Infinity,
    // so they go to the bottom when sorting in ascending order.
    function sortValue(name: string, column: SortColumn): number {
        const model = tip.models.get(name)!;
        if (column === "dim") return model.dim ?? Infinity;
        return timeOf(model.entries.get(column.slice("backend:".length)));
    }

    const rows = $derived.by(() => {
        const names = [...tip.models.keys()];
        const sign = sortAscending ? 1 : -1;
        const column = sortColumn;
        names.sort((a, b) => {
            if (column !== "name") {
                const [va, vb] = [sortValue(a, column), sortValue(b, column)];
                // Ties (including two Infinitys) fall through to sorting by name.
                if (va !== vb) return sign * (va < vb ? -1 : 1);
            }
            return sign * a.localeCompare(b);
        });
        return names;
    });

    function sortIndicator(column: SortColumn): string {
        if (sortColumn !== column) return "";
        return sortAscending ? " ▲" : " ▼";
    }

    function tooltip(entry: Entry, c: Comparison): string {
        const tipText = "time" in entry ? formatTime(entry.time) : "error";
        switch (c.kind) {
            case "no-base":
                return base === null ? "" : `tip: ${tipText}\nbase: (no result)`;
            case "both-error":
                return "errored in both tip and base";
            case "broke":
                return `tip: error\nbase: ${formatTime(c.baseTime)}`;
            case "fixed":
                return `tip: ${tipText}\nbase: error`;
            case "ratio":
                return (
                    `tip: ${tipText}\nbase: ${formatTime(c.baseTime)}\n` +
                    `change: ${formatChange(c.ratio)}`
                );
        }
    }
</script>

<table>
    <thead>
        <tr>
            <th class="model">
                <button onclick={() => setSort("name")}>Model{sortIndicator("name")}</button>
            </th>
            <th class="dim">
                <button onclick={() => setSort("dim")}>dim{sortIndicator("dim")}</button>
            </th>
            {#each tip.backends as backend}
                <th>
                    <button onclick={() => setSort(`backend:${backend}`)}>
                        {backend}{sortIndicator(`backend:${backend}`)}
                    </button>
                </th>
            {/each}
        </tr>
    </thead>
    <tbody>
        {#each rows as name}
            {@const model = tip.models.get(name)!}
            {@const baseModel = base?.models.get(name)}
            {@const primal = model.entries.get(PRIMAL)}
            {@const url = modelUrl(name)}
            <tr>
                <td class="model">
                    {#if url !== null}
                        <a href={url} target="_blank" rel="noreferrer">{name}</a>
                    {:else}
                        {name}
                    {/if}
                </td>
                <td class="dim">{model.dim ?? ""}</td>
                {#each tip.backends as backend}
                    {@const entry = model.entries.get(backend)}
                    {#if entry === undefined}
                        <td class="missing">—</td>
                    {:else}
                        {@const c = base
                            ? compare(entry, baseModel?.entries.get(backend))
                            : ({ kind: "no-base" } as Comparison)}
                        {@const err = "error" in entry ? entry.error : ""}
                        {@const tooltipText = err
                            ? `${err}${base ? "\n\n" + tooltip(entry, c) : ""}`
                            : tooltip(entry, c)}
                        <td
                            style:background-color={changeColour(c)}
                            class:has-tooltip={tooltipText !== ""}
                        >
                            {#if tooltipText !== ""}
                                <div class="tooltip">{tooltipText}</div>
                            {/if}
                            {#if "time" in entry}
                                {formatTime(entry.time)}
                                <!-- Always present in AD columns (even if empty) and of fixed
                                width, so that the times line up vertically. -->
                                {#if backend !== PRIMAL}
                                    <span class="ratio">
                                        {#if primal !== undefined && "time" in primal}
                                            ({(entry.time / primal.time).toFixed(1)}×)
                                        {/if}
                                    </span>
                                {/if}
                            {:else}
                                <span class="error">error</span>
                            {/if}
                        </td>
                    {/if}
                {/each}
            </tr>
        {/each}
    </tbody>
</table>

<style>
    table {
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    th,
    td {
        border: 1px solid var(--table-border);
        padding: 4px 12px;
        text-align: right;
        white-space: nowrap;
    }
    th {
        background-color: var(--table-header-bg);
    }
    th button {
        all: unset;
        cursor: pointer;
        font-weight: 600;
    }
    td {
        font-family: "Fira Code", monospace;
        font-size: 0.85rem;
    }
    .model {
        text-align: left;
    }
    td.model {
        font-family: inherit;
        font-size: 0.9rem;
    }
    .dim,
    .ratio,
    .missing {
        color: var(--text-secondary);
    }
    .ratio {
        font-size: 0.8em;
        display: inline-block;
        /* Wide enough for e.g. "(123.4×)" */
        min-width: 8ch;
        text-align: right;
    }
    .error {
        color: var(--error-color);
    }
    /* A CSS tooltip rather than `title`, because browsers delay showing `title`. */
    td.has-tooltip {
        position: relative;
        cursor: help;
    }
    .tooltip {
        display: none;
        position: absolute;
        top: 100%;
        right: 0;
        z-index: 10;
        width: max-content;
        max-width: 400px;
        padding: 6px 10px;
        text-align: left;
        white-space: pre-wrap;
        font-size: 0.8rem;
        color: var(--text-primary);
        background-color: var(--bg-secondary);
        border: 1px solid var(--table-border);
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        pointer-events: none;
    }
    td.has-tooltip:hover .tooltip {
        display: block;
    }
</style>
