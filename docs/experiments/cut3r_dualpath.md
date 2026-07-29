# CUT3R question-aware dual path

The canonical model is loaded directly from
`LLaVA-NeXT-Video-7B-Qwen2`.  The Journey9ni LoRA checkpoint must not be
passed as `model_name_or_path`, `model_base`, or an initialization source.

`train_cut3r_dualpath.sh` is the dedicated experiment wrapper. It sets the
SpatialStack checkpoint only as `--spatial_checkpoint`, disables legacy
`use_cut3r_spatialstack`, and enables the new independent side path.

The default position policy is `exact_index`: each CUT3R 27x27 patch receives
the position ID of the canonical patch with the same frame, row, and column.
The run fails if the canonical grid is not 27x27 or has fewer than 729 visual
patches per frame. This is intentional: no approximate mapping, duplicate
position IDs, or synthetic text-suffix positions are used by default.

The spatial branch consumes only pre-layer-0 prompt/nonvisual embeddings.
Teacher-forced answer tokens are excluded from branch conditioning but remain
eligible canonical writeback queries. During generation, processed dense
states are computed in prefill, stored next to the KV cache, and reused for
all decode tokens.

## Preflight gate

Before any full run, an HPC operator must run a target-GPU preflight with the
same 16-frame, 729-patch, dtype, per-device batch, checkpointing, and backend
as training. Record peak allocated/reserved memory and warmed median runtime
for global spatial attention, global text-only writeback, global all-token
writeback, complete prefill, complete forward/backward, greedy decode, and
beam decode.

Sweep `writeback_output_init_std` over `1e-5`, `1e-4`, and `1e-3`. Select the
smallest scale that keeps relative logit L2 deviation at most `1e-3`, produces
finite spatial/projector gradient RMS of at least `1e-8`, and gives each
spatial block/projector normalized gradient RMS at least `1e-3` of the median
downstream-LoRA RMS. Stop on OOM, nonfinite values, unsupported attention
backend, or less than 10% free device-memory headroom.

The `raw12` control bypasses the three spatial blocks and writes projected
final-layer CUT3R tokens directly. It shares writeback and training settings
with the processed branch, but is not a perfectly isolated maturation control:
the processed arm also uses progressive CUT3R levels 6/9/12.
