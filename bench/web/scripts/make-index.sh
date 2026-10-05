#!/usr/bin/env bash
# Writes `<dir>/index.json`, listing every run in <dir>. Needs `jq`.
#
#     scripts/make-index.sh path/to/runs
set -euo pipefail
dir="$1"
shopt -s nullglob
files=()
for f in "$dir"/*.json; do
    [[ "$(basename "$f")" == "index.json" ]] || files+=("$f")
done
if [[ ${#files[@]} -eq 0 ]]; then
    echo "[]" > "$dir/index.json"
else
    jq -n '[inputs | {file: (input_filename | split("/") | last), timestamp: .meta.timestamp, git_sha: .meta.git_sha, commit_date: .meta.commit_date, label: .meta.label}] | sort_by(.timestamp) | reverse' \
        "${files[@]}" > "$dir/index.json"
fi
