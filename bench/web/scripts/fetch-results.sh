#!/usr/bin/env bash
# Copies the saved benchmark runs from the `benchmark-results` branch into `public/runs`,
# so that `pnpm dev` can show them.
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch origin benchmark-results
rm -rf public/runs
mkdir -p public
# `git archive` only includes the current folder when run from a subfolder of the repo, so
# run it from the top of the repo.
git -C "$(git rev-parse --show-toplevel)" archive FETCH_HEAD runs | tar -x -C public
