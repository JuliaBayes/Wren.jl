<script lang="ts">
    import { formatTimestamp, isRelease, runLabel, shortSha } from "./format";
    import { commitUrl, runSha } from "./github";
    import type { IndexEntry, Run } from "./types";

    interface Props {
        label: string;
        index: IndexEntry[];
        /** Names of runs loaded from local files. */
        localFiles: string[];
        value: string;
        /** The selected run, once it has loaded. */
        run: Run | null;
        allowNone: boolean;
        onLoadFile: (file: File) => void;
    }
    let {
        label,
        index,
        localFiles,
        value = $bindable(),
        run,
        allowNone,
        onLoadFile,
    }: Props = $props();

    // Release runs (labelled e.g. "WrenCore-v0.2.0") are listed in their own group.
    let releases = $derived(index.filter((e) => isRelease(e)));
    let others = $derived(index.filter((e) => !isRelease(e)));

    let fileInput: HTMLInputElement;

    function onFileChange(e: Event) {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) onLoadFile(file);
        // Allow loading the same file again
        fileInput.value = "";
    }
</script>

<div class="picker">
    <label>
        <span class="label">{label}</span>
        <select bind:value>
            {#if allowNone}
                <option value="">(none)</option>
            {/if}
            {#if localFiles.length > 0}
                <optgroup label="Local files">
                    {#each localFiles as name}
                        <option value={`local:${name}`}>{name}</option>
                    {/each}
                </optgroup>
            {/if}
            {#each [["Releases", releases], ["Other runs", others]] as const as [groupLabel, entries]}
                {#if entries.length > 0}
                    <optgroup label={groupLabel}>
                        {#each entries as entry}
                            <option value={entry.file}>
                                {formatTimestamp(entry.timestamp)} {shortSha(entry.git_sha)} ({runLabel(entry)})
                            </option>
                        {/each}
                    </optgroup>
                {/if}
            {/each}
        </select>
    </label>
    <button onclick={() => fileInput.click()}>Load file…</button>
    <input
        type="file"
        accept=".json,application/json"
        bind:this={fileInput}
        onchange={onFileChange}
        hidden
    />
    <!-- Always present (even if empty), so that the page doesn't move when a run loads. -->
    <div class="info">
        {#if run !== null}
            {@const sha = runSha(run.meta)}
            {#if sha}
                <a href={commitUrl(sha)} target="_blank" rel="noreferrer">{shortSha(sha)}</a> ·
            {/if}
            Julia {run.meta.julia_version}
            <!-- Local files are already on disk, so there's nothing to download. -->
            {#if !value.startsWith("local:")}
                · <a href={`runs/${value}`} download={value}>Download JSON</a>
            {/if}
        {/if}
    </div>
</div>

<style>
    /* A grid, so that the info line lines up with the dropdown above it. */
    .picker {
        display: grid;
        grid-template-columns: 3.5em auto 1fr;
        align-items: center;
        column-gap: 8px;
    }
    label {
        display: contents;
    }
    .label {
        font-weight: 600;
    }
    button {
        justify-self: start;
    }
    select,
    button {
        font: inherit;
        font-size: 0.9rem;
        padding: 4px 8px;
        border: 1px solid var(--table-border);
        border-radius: 6px;
        background-color: var(--btn-bg);
        color: var(--btn-text);
    }
    /* A fixed width, so that the picker doesn't change size depending on its contents.
    Long labels are cut off, but are shown in full in the dropdown list. */
    select {
        width: 30em;
        max-width: 100%;
        text-overflow: ellipsis;
    }
    button {
        cursor: pointer;
    }
    button:hover {
        background-color: var(--btn-hover);
    }
    .info {
        grid-column: 2 / -1;
        /* Room for one line, even when empty */
        min-height: 1.6em;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
</style>
