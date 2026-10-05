#!/usr/bin/env bash
# Copies the saved benchmark runs from the `benchmark-results` branch into `public/runs`,
# so that `pnpm dev` can show them.
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch origin benchmark-results
rm -rf public/runs
mkdir -p public
git archive origin/benchmark-results runs | tar -x -C public
