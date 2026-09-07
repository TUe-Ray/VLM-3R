"""Four-rank NCCL all-reduce for smoke_environment.sbatch."""
import os

import torch
import torch.distributed as dist


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
value = torch.tensor([float(dist.get_rank() + 1)], device=f"cuda:{local_rank}")
dist.all_reduce(value)
expected = sum(range(1, dist.get_world_size() + 1))
if value.item() != expected:
    raise RuntimeError(f"NCCL all-reduce gave {value.item()}, expected {expected}")
print(
    f"rank={dist.get_rank()} local_rank={local_rank} "
    f"gpu={torch.cuda.get_device_name(local_rank)} nccl=ok"
)
dist.destroy_process_group()
