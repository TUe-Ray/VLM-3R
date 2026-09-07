#!/usr/bin/env bash
# Validate migrated VLM-3R assets without starting training.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "$SCRIPT_DIR/paths.env"

failures=0

fail() {
    printf '[FAIL] %s\n' "$*" >&2
    failures=$((failures + 1))
}

pass() {
    printf '[ OK ] %s\n' "$*"
}

require_dir() {
    local label="$1"
    local path="$2"
    if [[ -d "$path" ]]; then
        pass "$label: $path"
    else
        fail "$label missing: $path"
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ -f "$path" ]]; then
        pass "$label: $path"
    else
        fail "$label missing: $path"
    fi
}

require_writable_dir() {
    local label="$1"
    local path="$2"
    if [[ -d "$path" && -w "$path" ]]; then
        pass "$label writable: $path"
    else
        fail "$label is absent or not writable: $path"
    fi
}

report_feature_root() {
    local label="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        fail "$label feature root missing: $path"
        return
    fi

    local count
    count=$(find "$path" -type f -name '*.pt' -printf . | wc -c)
    local usage
    usage=$(du -sh "$path" | awk '{print $1}')
    if (( count == 0 )); then
        fail "$label feature root has no .pt files: $path (usage: $usage)"
    else
        pass "$label feature root: $path (.pt files: $count; usage: $usage)"
    fi
}

printf '%s\n' '== VLM-3R migration validation =='
printf 'REPO_DIR=%s\nVLM3R_ROOT=%s\n' "$REPO_DIR" "$VLM3R_ROOT"

require_dir 'repository' "$REPO_DIR"
require_dir 'base VLM: LLaVA-NeXT-Video-7B-Qwen2' "$LOCAL_MODEL_BASE"
require_dir 'vision tower: SigLIP' "$LOCAL_SIGLIP"

require_dir 'VSI-Bench data root / media root' "$VSI_BENCH_MEDIA_ROOT"
require_dir 'VSI-Bench SFT root' "$VSI_BENCH_TRAIN_ROOT"
require_file 'ScanNet SFT JSON' "$VSI_BENCH_TRAIN_ROOT/merged_qa_scannet_train.json"
require_file 'ScanNet++ SFT JSON' "$VSI_BENCH_TRAIN_ROOT/merged_qa_scannetpp_train.json"
require_file 'route-plan SFT JSON' "$VSI_BENCH_TRAIN_ROOT/merged_qa_route_plan_train.json"

report_feature_root 'CUT3R decoder 6' "$CUT3R_DEC6_ROOT"
report_feature_root 'CUT3R decoder 9' "$CUT3R_DEC9_ROOT"
report_feature_root 'CUT3R decoder 12' "$CUT3R_DEC12_ROOT"

require_dir 'CUT3R source submodule' "$REPO_DIR/third_party/CUT3R"
require_dir 'CUT3R source tree' "$REPO_DIR/third_party/CUT3R/src"
require_file 'CUT3R pretrained checkpoint' "$CUT3R_WEIGHTS_PATH"

require_writable_dir 'checkpoint root' "$TRAIN_SAVE_ROOT"
require_writable_dir 'Hugging Face cache' "$HF_HOME"
require_writable_dir 'Hugging Face datasets cache' "$HF_DATASETS_CACHE"
require_writable_dir 'Hugging Face hub cache' "$HUGGINGFACE_HUB_CACHE"
require_writable_dir 'log root' "$VLM3R_LOG_ROOT"

if (( failures > 0 )); then
    printf '\nMigration validation failed with %d missing or unusable required asset(s).\n' "$failures" >&2
    exit 1
fi

printf '\nMigration validation passed. Snellius smoke prerequisites are present.\n'
