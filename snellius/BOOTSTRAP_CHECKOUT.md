# Temporary bootstrap checkout

`/home/geusdd/VLM-3R` was created as a local bootstrap copy of the clean
Snellius checkout at `/home/geusdd/shuang/SpatialFocus`, commit
`858c761f6de72c8dda8ff5e2bc33e8d3c5b713f0` on `main`.

It is **not** the authoritative Leonardo working tree. Before training, replace
or synchronize it from the finalized Leonardo migration source while preserving
Leonardo's `.git`, current working tree, uncommitted modifications, and
initialized submodules. If an overwrite-style rsync uses `--delete`, exclude
`snellius/` or restore this directory immediately afterward.
