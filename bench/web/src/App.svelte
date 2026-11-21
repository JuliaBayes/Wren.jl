<script lang="ts">
    import { fetchIndex, fetchRun, readRunFile } from "./lib/load";
    import { NOISE_THRESHOLD } from "./lib/compare";
    import { shortSha } from "./lib/format";
    import { commitUrl, compareUrl, runSha } from "./lib/github";
    import type { IndexEntry, Run } from "./lib/types";
    import RunPicker from "./lib/RunPicker.svelte";
    import ResultsTable from "./lib/ResultsTable.svelte";
    import Manifest from "./lib/Manifest.svelte";
    import History from "./lib/History.svelte";

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
    // "compare" shows one or two runs as a table; "history" shows charts of all runs.
    let view: "compare" | "history" = $state(params.get("view") === "history" ? "history" : "compare");

    let tip = $state<Run | null>(null);
    let base = $state<Run | null>(null);
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
        if (view === "history") url.searchParams.set("view", view);
        else url.searchParams.delete("view");
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

    function compareRuns(tip: string, base: string) {
        tipKey = tip;
        baseKey = base;
        view = "compare";
        window.scrollTo(0, 0);
    }

    const tipSha = $derived(tip ? runSha(tip.meta) : null);
    const baseSha = $derived(base ? runSha(base.meta) : null);
</script>

<main>
    <div class="header">
        <h1>Wren benchmarks</h1>
        <button class="theme-toggle" onclick={toggleTheme} title={`${theme === "dark" ? "Dark" : "Light"} theme (click to switch)`}>
            <!-- Shows the current theme. -->
            {theme === "dark" ? "☾" : "☀"}
        </button>
    </div>

    <nav class="tabs">
        <button class:active={view === "compare"} onclick={() => (view = "compare")}>Compare runs</button>
        <button class:active={view === "history"} onclick={() => (view = "history")}>History</button>
    </nav>

    {#if view === "history"}
        <History {index} {theme} onCompare={compareRuns} />
    {:else}
        <div class="pickers">
            <RunPicker
                label="Tip"
                {index}
                {localFiles}
                bind:value={tipKey}
                run={tip}
                allowNone={false}
                onLoadFile={loadFile("tip")}
            />
            <RunPicker
                label="Base"
                {index}
                {localFiles}
                bind:value={baseKey}
                run={base}
                allowNone={true}
                onLoadFile={loadFile("base")}
            />
        </div>

        {#if error}
            <p class="error">{error}</p>
        {/if}

        {#if tip}
            {#if tipSha && baseSha && tipSha !== baseSha}
                <p class="diff">
                    <a href={compareUrl(baseSha, tipSha)} target="_blank" rel="noreferrer">
                        View diff on GitHub ({shortSha(baseSha)}...{shortSha(tipSha)})
                    </a>
                </p>
            {/if}
            <ul class="legend">
                <li>Times are the fastest of many evaluations of the log density (primal) or its gradient. Smaller is faster.</li>
                <li>Numbers in brackets are the gradient time divided by the primal time.</li>
                <li>
                    Model names link to their definitions
                    {#if tipSha}
                        at the tip commit (<a href={commitUrl(tipSha)} target="_blank" rel="noreferrer">{shortSha(tipSha)}</a>).
                    {:else}
                        on <code>main</code>, since the tip commit is unknown.
                    {/if}
                </li>
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
    .tabs {
        display: flex;
        gap: 4px;
        margin-bottom: 16px;
    }
    .tabs button {
        font: inherit;
        font-size: 0.9rem;
        padding: 4px 12px;
        border: 1px solid var(--table-border);
        border-radius: 6px;
        background-color: transparent;
        color: var(--text-secondary);
        cursor: pointer;
    }
    .tabs button:hover {
        background-color: var(--btn-hover);
    }
    .tabs button.active {
        background-color: var(--btn-bg);
        color: var(--btn-text);
        font-weight: 600;
    }
    .pickers {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 16px;
    }
    .legend {
        font-size: 0.9rem;
        padding-left: 18px;
    }
    .diff {
        font-size: 0.9rem;
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
