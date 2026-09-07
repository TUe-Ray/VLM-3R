#!/usr/bin/env bash
# Validate the complete manifest-scoped bundle after it lands on Snellius.
set -Eeuo pipefail

REPO_DIR="/home/geusdd/VLM-3R"
ROOT="/scratch-shared/geusdd/VLM3R"
REL="$REPO_DIR/migration/snellius/relative_lists"
EXPECTED_BRANCH="feat/new_design"
EXPECTED_HEAD="02ce5b541267369f7c9a61c5f0e6557a94d44b2d"
EXPECTED_CUT3R="51244364af3566d6473559f71a81b4accc75c424"

check_list() {
    local name="$1"
    local root="$2"
    local list="$3"
    local count="$4"
    local bytes="$5"
    /usr/bin/python3 - "$name" "$root" "$list" "$count" "$bytes" <<'PY'
import pathlib
import sys

name, root, list_path, expected_count, expected_bytes = sys.argv[1:]
root = pathlib.Path(root)
entries = pathlib.Path(list_path).read_text(encoding="utf-8").splitlines()
expected_count = int(expected_count)
expected_bytes = int(expected_bytes)
assert len(entries) == expected_count, f"{name}: {len(entries)} entries"
assert len(set(entries)) == expected_count, f"{name}: duplicate entries"
paths = [root / item for item in entries]
missing = [path for path in paths if not path.is_file()]
assert not missing, f"{name}: {len(missing)} missing; first={missing[:1]}"
actual = sum(path.stat().st_size for path in paths)
assert actual == expected_bytes, f"{name}: {actual} != {expected_bytes} bytes"
print(f"[ OK ] {name}: {len(paths)} files, {actual} bytes")
PY
}

check_tree() {
    local name="$1"
    local root="$2"
    local count="$3"
    local bytes="$4"
    /usr/bin/python3 - "$name" "$root" "$count" "$bytes" <<'PY'
import os
import pathlib
import sys

name, root, expected_count, expected_bytes = sys.argv[1:]
root = pathlib.Path(root)
assert root.is_dir(), f"{name}: missing directory {root}"
files = [pathlib.Path(d) / f for d, _, fs in os.walk(root) for f in fs]
actual = sum(path.stat().st_size for path in files)
assert len(files) == int(expected_count), f"{name}: {len(files)} != {expected_count} files"
assert actual == int(expected_bytes), f"{name}: {actual} != {expected_bytes} bytes"
print(f"[ OK ] {name}: {len(files)} files, {actual} bytes")
PY
}

check_sha() {
    local expected="$1"
    local path="$2"
    local actual
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
        printf '[FAIL] SHA-256 mismatch: %s\n' "$path" >&2
        exit 1
    }
    printf '[ OK ] SHA-256: %s\n' "$path"
}

[[ -d "$REPO_DIR" && -d "$ROOT" && -d "$REL" ]]
[[ "$(git -C "$REPO_DIR" branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
CUT3R_ACTUAL=$(git -C "$REPO_DIR" submodule status -- third_party/CUT3R | awk '{print $1}' | tr -d '+-')
[[ "$CUT3R_ACTUAL" == "$EXPECTED_CUT3R" ]]
printf '[ OK ] repository: %s @ %s; CUT3R %s\n' "$EXPECTED_BRANCH" "$EXPECTED_HEAD" "$EXPECTED_CUT3R"

check_tree 'LLaVA' "$ROOT/models/LLaVA-NeXT-Video-7B-Qwen2" 37 16074683170
check_tree 'SigLIP' "$ROOT/models/siglip-so400m-patch14-384" 8 3515154696
check_sha 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 "$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth"

check_list 'training JSON' "$ROOT/data/vlm3r" "$REL/training_jsons.rel" 3 129175285
check_list 'training media' "$ROOT/data/vlm3r" "$REL/training_media.rel" 2405 513850287837
check_list 'training dec6' "$ROOT/spatial_features/cut3r/dec6" "$REL/training_dec6.rel" 2405 86298745486
check_list 'training dec9' "$ROOT/spatial_features/cut3r/dec9" "$REL/training_dec9.rel" 2405 86298745486
check_list 'training dec12' "$ROOT/spatial_features/cut3r/dec12" "$REL/training_dec12.rel" 2405 86297360206
check_list 'eval metadata' "$ROOT/data/vsibench" "$REL/eval_metadata.rel" 2 175344
check_list 'eval media' "$ROOT/hf_cache/vsibench" "$REL/eval_media.rel" 288 3728068588
check_list 'eval dec6' "$ROOT/spatial_features/cut3r/dec6" "$REL/eval_dec6.rel" 288 10334299224
check_list 'eval dec9' "$ROOT/spatial_features/cut3r/dec9" "$REL/eval_dec9.rel" 288 10334299224
check_list 'eval dec12' "$ROOT/spatial_features/cut3r/dec12" "$REL/eval_dec12.rel" 288 10334151768

check_sha 6cf0368fc34124cd9a3c60077a84704f04a0829bf9ee8296d35bb8242fd9df1e "$ROOT/data/vlm3r/VLM-3R-DATA/vsibench_train/merged_qa_scannet_train.json"
check_sha 00d9de17925ffdf530941d621e70c4855d3f329ad51063ec6648bc8160b135cc "$ROOT/data/vlm3r/VLM-3R-DATA/vsibench_train/merged_qa_scannetpp_train.json"
check_sha dbefc6f768614c10ee839bb35786f9f4df12b92691875e7482fa47ded01ba93b "$ROOT/data/vlm3r/VLM-3R-DATA/vsibench_train/merged_qa_route_plan_train.json"

/usr/bin/python3 - "$REL/training_media.rel" "$REL/eval_media.rel" <<'PY'
import pathlib
import sys

def keys(path):
    return {(item.split('/')[0], pathlib.Path(item).stem) for item in pathlib.Path(path).read_text().splitlines()}

train = keys(sys.argv[1])
evaluation = keys(sys.argv[2])
assert len(train) == 2405 and len(evaluation) == 288
assert train.isdisjoint(evaluation), f"overlap: {sorted(train & evaluation)[:3]}"
print('[ OK ] training/evaluation media scene sets are disjoint')
PY

printf '\nTarget bundle validation passed.\n'
