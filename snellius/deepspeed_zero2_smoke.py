"""One-node ZeRO-2 optimizer smoke; intentionally unrelated to VLM SFT."""

import os

import deepspeed
import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")

    model = torch.nn.Linear(16, 16, bias=False).cuda()
    config = {
        "train_batch_size": dist.get_world_size(),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
        "zero_optimization": {"stage": 2},
    }
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    data = torch.randn(1, 16, device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    loss = engine(data).float().square().mean()
    engine.backward(loss)
    engine.step()
    dist.barrier()
    print(f"rank={dist.get_rank()} deepspeed_zero2=ok loss={loss.item():.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
