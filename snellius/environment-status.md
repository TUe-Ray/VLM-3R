# Snellius environment status

The active environment is `/home/geusdd/.conda/envs/vlm3r-snellius` with
Python 3.10.14, PyTorch 2.1.1+cu121, Transformers 4.40.0.dev0, DeepSpeed
0.14.4, Accelerate 0.29.1, and editable `llava` and `lmms_eval` installs.

Do not activate it by name after loading the Snellius Miniconda module: that
can put the module base Python before the environment interpreter. Active
wrappers instead prepend the absolute environment `bin` directory to `PATH`.

FlashAttention 2.7.1.post1 and CUT3R CUROPE were rebuilt from Snellius source
and passed functional CUDA checks in job 26432129. The same job passed
four-rank NCCL and DeepSpeed ZeRO-2 checks.

Additional packages installed for the offline training/evaluation path are
`hf-transfer==0.1.9`, `protobuf==6.33.6`, and `tyro==1.0.10`. `pip check`
still reports absent development/optional packages (black, ftfy, isort,
openai, pre-commit, pybind11, and yt-dlp) plus optional-version conflicts in
the combined historical requirements. The real offline training and
VSI-Bench generation smokes pass without those extras; do not change the
working runtime merely to make the union of old requirement files consistent.
