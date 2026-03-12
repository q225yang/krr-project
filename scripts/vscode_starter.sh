(base) [qyang129@sol-login05:~]$ cat vscode_starter.sh 
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./vscode_starter.sh 0-4    # 4 hours
#   ./vscode_starter.sh 0-12   # 12 hours
#   ./vscode_starter.sh 1-0    # 1 day
#
# If omitted, defaults to 0-4.

TIME="${1:-0-4}"

# Validate Slurm-style time format D-H (days-hours), e.g. 0-4, 1-0, 2-12
if [[ ! "$TIME" =~ ^[0-9]+-[0-9]+$ ]]; then
  echo "Error: time must be in D-H format like 0-4, 0-12, 1-0" >&2
  exit 1
fi

DAYS="${TIME%-*}"
HOURS="${TIME#*-}"

# Basic sanity checks
if (( HOURS < 0 || HOURS > 23 )); then
  echo "Error: hours must be 0..23 (got $HOURS). Use days for longer durations, e.g. 1-0 for 24h." >&2
  exit 1
fi

echo "Starting VS Code session with time limit: $TIME (days-hours)"
exec vscode -p general --gres=gpu:1 -t "$TIME"