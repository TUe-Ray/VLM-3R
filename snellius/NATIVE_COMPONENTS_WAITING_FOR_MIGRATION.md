# NATIVE_COMPONENTS_WAITING_FOR_MIGRATION

No compiled artifact from Leonardo should be copied to Snellius. Compile each
component from source in `vlm3r-snellius` only after the authoritative working
tree, submodules, and exact package/CUDA pins have arrived.

1. **CUT3R / CUROPE** — `third_party/CUT3R` is currently an uninitialized git
   submodule (gitlink `51244364af3566d6473559f71a81b4accc75c424`). The expected
   CUROPE source location is `third_party/CUT3R/src/croco/models/curope`, but
   it is unavailable until the submodule content is migrated or initialized.
2. **FlashAttention** — a CUDA extension. The checked-in requirements conflict
   on its version, so defer its build until `environment-leonardo.yml` or
   `pip-freeze-leonardo.txt` selects one version and the PyTorch CUDA build is
   known.
3. **DeepSpeed CUDA ops** — DeepSpeed 0.14.4 is pinned in the broad repository
   requirements. Build optional CUDA ops only after PyTorch, its CUDA build,
   compiler module, and NCCL behavior have passed the environment smoke job.

Before compiling, capture `torch.__version__`, `torch.version.cuda`,
`nvcc --version`, and `nvidia-smi` on the target GPU partition. Use a CUDA
toolkit that matches the selected PyTorch CUDA build; do not combine a copied
Leonardo `.so` with Snellius libraries.
