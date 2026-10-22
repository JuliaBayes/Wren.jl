export type Version = string | null;

export type PackageChange =
    | { kind: "unchanged"; version: Version }
    | { kind: "changed"; old: Version; new: Version }
    | { kind: "added"; new: Version }
    | { kind: "removed"; old: Version };

/**
 * Compare the package versions in two manifests. If there is no base manifest, every
 * package is reported as unchanged. Sorted by package name.
 */
export function diffManifests(
    tip: Map<string, Version>,
    base: Map<string, Version> | null,
): [string, PackageChange][] {
    const names = new Set([...tip.keys(), ...(base?.keys() ?? [])]);
    return [...names]
        .sort((a, b) => a.localeCompare(b))
        .map((name): [string, PackageChange] => {
            const inTip = tip.has(name);
            const t = tip.get(name) ?? null;
            if (base === null) return [name, { kind: "unchanged", version: t }];
            const inBase = base.has(name);
            const b = base.get(name) ?? null;
            if (inTip && !inBase) return [name, { kind: "added", new: t }];
            if (!inTip && inBase) return [name, { kind: "removed", old: b }];
            if (t === b) return [name, { kind: "unchanged", version: t }];
            return [name, { kind: "changed", old: b, new: t }];
        });
}
