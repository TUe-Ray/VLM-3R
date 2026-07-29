# Dual-path controlled experiment results

| Run | Seed | Spatial attention | Writeback scope | Writeback visibility | Evidence | Init std | VSI Avg | Object Count | Abs. Distance | Object Size | Room Size | Rel. Distance | Rel. Direction | Route Plan | Appearance Order |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2D base |  | disabled | — | — | — | — |  |  |  |  |  |  |  |  |  |
| SpatialStack 6/9/12 |  | — | — | — | legacy | — |  |  |  |  |  |  |  |  |  |
| Raw layer-12 |  | — | all_tokens | frame_local | final dense |  |  |  |  |  |  |  |  |  |  |
| A |  | global | all_tokens | frame_local | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |
| B |  | frame_local | all_tokens | frame_local | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |
| C |  | global | all_tokens | global | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |
| D |  | global | text_only | global | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |

Record the target-GPU preflight report and checkpoint source alongside every
row. The raw control changes both maturation and progressive feature-level
usage relative to the processed arm; do not interpret it as a perfectly
isolated maturation ablation.
