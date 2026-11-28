#!/usr/bin/env bash
# Runs a Julia script in this folder, saves its output, and sends a phone notification
# through ntfy.sh when it finishes (whether or not it succeeded).
#
# Usage, from this folder:
#
#     NTFY_TOPIC=<your-topic> ./run_and_notify.sh diagnose.jl diagnose.txt
#
# The topic is read from NTFY_TOPIC rather than written here, because anyone who knows a
# topic name can read and send its messages, and this file is public. Without NTFY_TOPIC
# the script still runs, it just doesn't send a notification.

set -u

if [ $# -ne 2 ]; then
    echo "usage: $0 <script.jl> <output.txt>" >&2
    exit 2
fi
script=$1
output=$2

start=$(date +%s)
julia --project=. --threads=auto "$script" 2>&1 | tee "$output"
status=${PIPESTATUS[0]}
minutes=$(( ($(date +%s) - start) / 60 ))

if [ -n "${NTFY_TOPIC:-}" ]; then
    curl -s -d "$script finished with exit code $status after $minutes min" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null
fi
exit "$status"
