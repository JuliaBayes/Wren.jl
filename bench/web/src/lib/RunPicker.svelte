<script lang="ts">
    import { formatTimestamp, isRelease, runLabel, shortSha } from "./format";
    import type { IndexEntry } from "./types";

    interface Props {
        label: string;
        index: IndexEntry[];
        /** Names of runs loaded from local files. */
        localFiles: string[];
        value: string;
        allowNone: boolean;
        onLoadFile: (file: File) => void;
    }
    let {
        label,
        index,
        localFiles,
        value = $bindable(),
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
</div>

<style>
    .picker {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .label {
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
        width: 3em;
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
    select {
        min-width: 260px;
    }
    button {
        cursor: pointer;
    }
    button:hover {
        background-color: var(--btn-hover);
    }
</style>
