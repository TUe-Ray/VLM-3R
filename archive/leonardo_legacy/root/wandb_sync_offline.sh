#!/bin/bash

WANDB_ROOT="${WANDB_ROOT:-$WORK/wandb/wandb}"   # 指到「含有 offline-run-* 的那層」
SYNC_INTERVAL="${SYNC_INTERVAL:-500}"
ACTIVE_MINUTES="${ACTIVE_MINUTES:-5}"

export PATH="$WORK/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vlm3r
# Parse command line arguments
SYNC_ALL=false
if [ "$1" == "-all" ]; then
    SYNC_ALL=true
    echo "Mode: Syncing ALL runs regardless of modification time"
fi

# Load wandb credentials
if [ -f "$HOME/.wandb_env" ]; then
    source "$HOME/.wandb_env"
else
    echo "Error: ~/.wandb_env not found."
    exit 1
fi

# Ensure API key is set
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set."
    exit 1
fi

# Check if base directory exists
if [ ! -d "$WANDB_ROOT" ]; then
    echo "Error: Base directory $WANDB_ROOT does not exist."
    exit 1
fi

# Change to base directory
cd "$WANDB_ROOT" || exit 1

# Handle Ctrl+C gracefully
trap 'echo ""; echo "Stopping wandb sync..."; exit 0' INT TERM

echo "Starting continuous wandb sync from $WANDB_ROOT"
echo "Sync interval: ${SYNC_INTERVAL} seconds"
echo "Press Ctrl+C to stop"
echo "======================================"
echo ""

# If no matching dirs, glob expands to nothing (instead of literal pattern)
shopt -s nullglob

# Continuous loop
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Starting sync iteration..."
    echo "--------------------------------------"

    synced_runs=()

    # ---- Sync based on mode ----
    if [ "$SYNC_ALL" = true ]; then
        # Sync all offline runs
        mapfile -d '' active_runs < <(
            find "$WANDB_ROOT" -maxdepth 1 -mindepth 1 -type d -name "offline-run-*" -print0
        )
        
        if [ ${#active_runs[@]} -eq 0 ]; then
            echo "No offline runs found."
        fi
    else
        # Only sync runs updated recently (last ACTIVE_MINUTES)
        mapfile -d '' active_runs < <(
            find "$WANDB_ROOT" -maxdepth 1 -mindepth 1 -type d -name "offline-run-*" -print0 | xargs -0 -I {} find {} -type f -mmin "-$ACTIVE_MINUTES" -print0 | xargs -0 dirname | sort -u | xargs -I {} printf '%s\0' {}
        )

        if [ ${#active_runs[@]} -eq 0 ]; then
            echo "No active runs updated in last ${ACTIVE_MINUTES} minutes."
        fi
    fi

    # Sync the runs
    for run_dir in "${active_runs[@]}"; do
        if [ -d "$run_dir" ]; then
            echo "Syncing run: $(basename "$run_dir")"

            set +e
            wandb sync "$run_dir" 2>&1 | grep -v "^wandb:"
            sync_exit_code=${PIPESTATUS[0]}
            set -e

            if [ $sync_exit_code -eq 0 ]; then
                synced_runs+=("$(basename "$run_dir")")
            else
                echo "  ✗ Failed (exit code: $sync_exit_code)"
            fi
        fi
    done

    if [ ${#synced_runs[@]} -gt 0 ]; then
        echo "Synced runs: ${synced_runs[*]}"
    fi

    echo "--------------------------------------"
    echo "[$TIMESTAMP] Waiting ${SYNC_INTERVAL} seconds..."
    echo ""
    sleep "$SYNC_INTERVAL"
done
