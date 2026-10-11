<script lang="ts">
    import { diffManifests, type Version } from "./manifest";
    import type { Run } from "./types";

    interface Props {
        tip: Run;
        base: Run | null;
    }
    let { tip, base }: Props = $props();

    const rows = $derived(
        tip.manifest === null ? [] : diffManifests(tip.manifest, base?.manifest ?? null),
    );
    const numChanges = $derived(rows.filter(([, c]) => c.kind !== "unchanged").length);
    const comparing = $derived(base !== null && base.manifest !== null);

    let onlyChanges = $state(false);
    const shownRows = $derived(
        onlyChanges ? rows.filter(([, c]) => c.kind !== "unchanged") : rows,
    );

    function v(version: Version): string {
        return version ?? "(no version)";
    }
</script>

{#if tip.manifest === null}
    <p class="note">The tip run has no manifest.</p>
{:else}
    {#if base !== null && base.manifest === null}
        <p class="note">The base run has no manifest, so changes can't be shown.</p>
    {/if}
    {#if comparing}
        <p class="note">
            {numChanges} package{numChanges === 1 ? "" : "s"} changed since base.
            <span class="changed">changed</span>
            <span class="added">added</span>
            <span class="removed">removed</span>
            <label>
                <input type="checkbox" bind:checked={onlyChanges} /> Only show changes
            </label>
        </p>
    {/if}
    <table>
        <tbody>
            {#each shownRows as [name, c]}
                <tr>
                    <td>{name}</td>
                    <td class="version">
                        {#if c.kind === "unchanged"}
                            {v(c.version)}
                        {:else if c.kind === "changed"}
                            <span class="changed">{v(c.old)} ⇒ {v(c.new)}</span>
                        {:else if c.kind === "added"}
                            <span class="added">(newly added) {v(c.new)}</span>
                        {:else}
                            <span class="removed">(removed) {v(c.old)}</span>
                        {/if}
                    </td>
                </tr>
            {/each}
        </tbody>
    </table>
{/if}

<style>
    table {
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    td {
        border: 1px solid var(--table-border);
        padding: 2px 12px;
    }
    td.version {
        font-family: "Fira Code", monospace;
    }
    .note {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    .note span {
        margin-left: 6px;
    }
    label {
        margin-left: 16px;
    }
    .changed,
    .added,
    .removed {
        padding: 0 4px;
        border-radius: 3px;
    }
    .changed {
        background-color: color-mix(in srgb, var(--changed) 50%, transparent);
    }
    .added {
        background-color: color-mix(in srgb, var(--faster) 50%, transparent);
    }
    .removed {
        background-color: color-mix(in srgb, var(--slower) 50%, transparent);
    }
</style>
