<script lang="ts">
    import { changeColour, compare, type Comparison } from "./compare";
    import { formatChange, formatTime } from "./format";
    import type { Entry, Run } from "./types";

    interface Props {
        tip: Run;
        base: Run | null;
    }
    let { tip, base }: Props = $props();

    // The backend that AD timings are divided by to get the "× primal" ratio.
    const PRIMAL = "primal";

    // Sorting: by model name (null) or by the tip time of a backend.
    let sortBackend: string | null = $state(null);
    let sortAscending = $state(true);

    function setSort(backend: string | null) {
        if (sortBackend === backend) {
            sortAscending = !sortAscending;
        } else {
            sortBackend = backend;
            sortAscending = true;
        }
    }

    function timeOf(entry: Entry | undefined): number {
        return entry !== undefined && "time" in entry ? entry.time : Infinity;
    }

    const rows = $derived.by(() => {
        const names = [...tip.models.keys()];
        const sign = sortAscending ? 1 : -1;
        if (sortBackend === null) {
            names.sort((a, b) => sign * a.localeCompare(b));
        } else {
            const key = sortBackend;
            names.sort(
                (a, b) =>
                    sign *
                    (timeOf(tip.models.get(a)!.entries.get(key)) -
                        timeOf(tip.models.get(b)!.entries.get(key))),
            );
        }
        return names;
    });

    function sortIndicator(backend: string | null): string {
        if (sortBackend !== backend) return "";
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
                    `change: ${formatChange(c.ratio)} (${c.ratio.toFixed(2)}×)`
                );
        }
    }
</script>

<table>
    <thead>
        <tr>
            <th class="model">
                <button onclick={() => setSort(null)}>Model{sortIndicator(null)}</button>
            </th>
            <th class="dim">dim</th>
            {#each tip.backends as backend}
                <th>
                    <button onclick={() => setSort(backend)}>
                        {backend}{sortIndicator(backend)}
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
            <tr>
                <td class="model">{name}</td>
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
                        <td
                            style:background-color={changeColour(c)}
                            title={err
                                ? `${err}${base ? "\n\n" + tooltip(entry, c) : ""}`
                                : tooltip(entry, c)}
                        >
                            {#if "time" in entry}
                                {formatTime(entry.time)}
                                {#if backend !== PRIMAL && primal !== undefined && "time" in primal}
                                    <span class="ratio">
                                        ({(entry.time / primal.time).toFixed(1)}×)
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
    }
    .error {
        color: var(--error-color);
    }
    td[title]:not([title=""]) {
        cursor: help;
    }
</style>
