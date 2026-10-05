<script lang="ts">
    import { fetchIndex, fetchRun, readRunFile } from "./lib/load";
    import { NOISE_THRESHOLD } from "./lib/compare";
    import { formatTimestamp, runLabel, shortSha } from "./lib/format";
    import type { IndexEntry, Run } from "./lib/types";
    import RunPicker from "./lib/RunPicker.svelte";
    import ResultsTable from "./lib/ResultsTable.svelte";
    import Manifest from "./lib/Manifest.svelte";

    let theme = $state(document.documentElement.getAttribute("data-theme") ?? "light");
    function toggleTheme() {
        theme = theme === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("theme", theme);
    }

    // Runs are identified by a key: either a file name in `runs/index.json`, or
    // `local:<name>` for a file loaded from disk. "" means no run selected.
    let index: IndexEntry[] = $state([]);
    let localRuns: Map<string, Run> = $state(new Map());
    const localFiles = $derived([...localRuns.values()].map((r) => r.source));

    const params = new URLSearchParams(window.location.search);
    let tipKey = $state(params.get("tip") ?? "");
    let baseKey = $state(params.get("base") ?? "");

    let tip: Run | null = $state(null);
    let base: Run | null = $state(null);
    let error: string | null = $state(null);

    fetchIndex().then((entries) => {
        // Newest first
        index = entries.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
        if (tipKey === "" && index.length > 0) tipKey = index[0].file;
    });

    async function getRun(key: string): Promise<Run | null> {
        if (key === "") return null;
        if (key.startsWith("local:")) return localRuns.get(key) ?? null;
        return fetchRun(key);
    }

    // Load the selected runs whenever the selection changes.
    $effect(() => {
        const [t, b] = [tipKey, baseKey];
        error = null;
        Promise.all([getRun(t), getRun(b)])
            .then(([tipRun, baseRun]) => {
                // Ignore results if the selection changed while loading
                if (t !== tipKey || b !== baseKey) return;
                tip = tipRun;
                base = baseRun;
            })
            .catch((e) => (error = String(e)));
    });

    // Keep the URL in sync, so that comparisons can be linked. Local files can't be
    // linked to, so they're left out.
    $effect(() => {
        const url = new URL(window.location.href);
        for (const [name, key] of [["tip", tipKey], ["base", baseKey]]) {
            if (key === "" || key.startsWith("local:")) url.searchParams.delete(name);
            else url.searchParams.set(name, key);
        }
        history.replaceState(null, "", url);
    });

    function loadFile(which: "tip" | "base") {
        return async (file: File) => {
            try {
                const run = await readRunFile(file);
                const key = `local:${run.source}`;
                localRuns = new Map(localRuns).set(key, run);
                if (which === "tip") tipKey = key;
                else baseKey = key;
            } catch (e) {
                error = String(e);
            }
        };
    }

    function describe(run: Run): string {
        return `${run.source} · ${shortSha(run.meta.git_sha)} · ${formatTimestamp(run.meta.timestamp)} · Julia ${run.meta.julia_version}`;
    }
</script>

<main>
    <div class="header">
        <h1>Wren benchmarks</h1>
        <button class="theme-toggle" onclick={toggleTheme} title="Toggle theme">
            {theme === "dark" ? "☀" : "☾"}
        </button>
    </div>

    <div class="pickers">
        <RunPicker
            label="Tip"
            {index}
            {localFiles}
            bind:value={tipKey}
            allowNone={false}
            onLoadFile={loadFile("tip")}
        />
        <RunPicker
            label="Base"
            {index}
            {localFiles}
            bind:value={baseKey}
            allowNone={true}
            onLoadFile={loadFile("base")}
        />
    </div>

    {#if error}
        <p class="error">{error}</p>
    {/if}

    {#if tip}
        <ul class="run-info">
            <li><b>Tip:</b> <b>{runLabel(tip.meta)}</b> · {describe(tip)}</li>
            {#if base}
                <li><b>Base:</b> <b>{runLabel(base.meta)}</b> · {describe(base)}</li>
            {/if}
        </ul>
        <ul class="legend">
            <li>Times are the median of many evaluations of the log density (primal) or its gradient. Smaller is faster.</li>
            <li>Numbers in brackets are the gradient time divided by the primal time.</li>
            {#if base}
                <li>
                    <span class="swatch faster"></span> faster than base,
                    <span class="swatch slower"></span> slower than base
                    (changes under {NOISE_THRESHOLD * 100}% are not coloured).
                    Hover over a cell to see the base time.
                </li>
            {/if}
        </ul>
        <ResultsTable {tip} {base} />

        <h2>Manifest</h2>
        <Manifest {tip} {base} />
    {:else if !error}
        <p>No runs found. Load a results file with the button above.</p>
    {/if}
</main>

<style>
    main {
        padding: 10px 40px 50px;
        max-width: 1200px;
    }
    .header {
        display: flex;
        align-items: center;
        gap: 16px;
        border-bottom: 2px solid var(--table-border);
        margin-bottom: 16px;
    }
    h1 {
        margin: 8px 0;
        font-size: 1.75rem;
    }
    button.theme-toggle {
        margin-left: auto;
        background: transparent;
        color: var(--text-secondary);
        border: 1px solid var(--table-border);
        width: 34px;
        height: 34px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
    }
    .pickers {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 16px;
    }
    .run-info,
    .legend {
        font-size: 0.9rem;
        padding-left: 18px;
    }
    .legend {
        color: var(--text-secondary);
    }
    .swatch {
        display: inline-block;
        width: 0.9em;
        height: 0.9em;
        border-radius: 2px;
        vertical-align: middle;
    }
    .swatch.faster {
        background-color: var(--faster);
    }
    .swatch.slower {
        background-color: var(--slower);
    }
    .error {
        color: var(--error-color);
    }
</style>
